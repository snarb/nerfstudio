"""Source-bound CPU final encoding. RGB masters and frozen shaders stay unchanged."""
from pathlib import Path
import fcntl, hashlib, json, os, re, subprocess, uuid
import numpy as np
from PIL import Image

class AwaitEncodingOwnership(RuntimeError):
    pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def command(ffmpeg,pattern,count,output):
    return [str(ffmpeg),'-hide_banner','-loglevel','error','-y','-framerate','24',
        '-start_number','0','-i',str(pattern),'-frames:v',str(count),'-r','24',
        '-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709',
        '-color_range','tv','-colorspace','bt709','-color_primaries','bt709',
        '-color_trc','iec61966-2-1','-c:v','libx264','-preset','slow','-crf','15',
        '-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(output)]

def probe(ffprobe,path,count):
    value=json.loads(subprocess.check_output([str(ffprobe),'-v','error',
        '-show_streams','-show_format','-of','json',str(path)],text=True))
    streams=[s for s in value['streams']if s['codec_type']=='video']
    if len(streams)!=1:raise ValueError('Exactly one video stream required')
    expected=dict(codec_name='h264',width=3840,height=2024,pix_fmt='yuv420p',
        color_range='tv',color_space='bt709',color_primaries='bt709',
        color_transfer='iec61966-2-1',r_frame_rate='24/1',avg_frame_rate='24/1',
        nb_frames=str(count))
    for name,match in expected.items():
        if streams[0].get(name)!=match:raise ValueError('Encoded stream mismatch '+name)
    if abs(float(value['format']['duration'])-count/24)>1/24000:
        raise ValueError('Encoded duration mismatch')
    return value

def decode_check(ffmpeg,path,count,frame_root,variant):
    indices=sorted(set([0,count//2,count-1]));selection='+'.join('eq(n\\,%d)'%i for i in indices)
    raw=subprocess.check_output([str(ffmpeg),'-v','error','-err_detect','explode',
        '-i',str(path),'-vf','select='+selection,'-vsync','0','-pix_fmt','rgb24',
        '-threads','2','-f','rawvideo','-'])
    expected=len(indices)*2024*3840*3
    if len(raw)!=expected:raise ValueError('Decoded RGB raster/frame count mismatch')
    frames=np.frombuffer(raw,np.uint8).reshape(len(indices),2024,3840,3);stats=[]
    for index,decoded in zip(indices,frames):
        source=Path(frame_root)/f'{index:04d}'/(variant+'.png')
        original=np.asarray(Image.open(source).convert('RGB'))
        if original.shape!=decoded.shape:raise ValueError('PNG/decoded raster mismatch')
        error=decoded.astype(np.float64)-original;absolute=abs(error);mse=float(np.mean(error**2))
        stats.append(dict(index=index,PNG_sha256=sha(source),decoded_RGB_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
            RGB_code_MAE=float(absolute.mean()),RGB_code_P99=float(np.quantile(absolute,.99)),
            RGB_PSNR=float(10*np.log10(255**2/mse))if mse else None,
            lossy=True,metric_is_codec_RGB_parity_not_model_quality=True))
    return dict(decoded_frames=indices,no_resize=True,decode_all_to_last_frame_succeeded=True,measurements=stats)

class Encoder:
    def __init__(self,policy,guard,*,event=None):
        self.policy=policy;self.guard=guard;self.event=event or(lambda name,value:None)

    def sources(self,spec,path,count):
        request=read(path);root=Path(spec['output']);masters={};completions={}
        if request.get('output_wh')!=[3840,2024]or len(request['records'])!=120:
            raise ValueError('Expected qualified120-record4K path')
        if not(request.get('version')==4 and request.get('paused_GT_authorized')is True):
            raise ValueError('Expected authorized V4 paused-GT path')
        path_digest=sha(path)
        for i in range(count):
            directory=root/'frames'/f'{i:04d}';p=directory/'complete.json';receipt=read(p)
            if receipt['binding']['path_sha256']!=path_digest:raise ValueError('Mixed path campaign')
            if spec.get('_spec_path')and receipt['binding']['spec_sha256']!=sha(spec['_spec_path']):
                raise ValueError('Mixed source render spec')
            if receipt['binding'].get('record')!=request['records'][i]:raise ValueError('Mixed source record')
            if set(receipt['images'])!={'lookcloser.png','hybrid.png'}:raise ValueError('Both PNG masters required')
            completions[str(p)]=sha(p)
            for name,value in receipt['images'].items():
                image=directory/name
                if sha(image)!=value:raise ValueError('Frame hash changed')
                with Image.open(image)as im:
                    if im.size!=(3840,2024):raise ValueError('PNG dimensions changed')
                masters[str(image)]=value
        return dict(path_sha256=path_digest,count=count,fps=24,output_wh=[3840,2024],
            PNG_masters_sha256=masters,source_completion_sha256=completions,
            policy_sha256=digest(self.policy),profile_sha256=self.policy['profile_sha256'],encoding_adapter_sha256=sha(__file__))

    def unchanged(self,binding):
        for key in ['PNG_masters_sha256','source_completion_sha256']:
            for path,value in binding[key].items():
                if sha(path)!=value:raise ValueError('Source changed during encoding')
        self.guard()

    def inspect(self,path,count,frame_root,variant):
        return dict(ffprobe=probe(self.policy['ffprobe'],path,count),
            decode=decode_check(self.policy['ffmpeg'],path,count,frame_root,variant))

    def one(self,root,variant,count,binding,lock_fd):
        suffix='neural3s'if count==72 else'native_GT2s'
        output=root/f'Marshawn1660_3s_24fps_{variant}_{suffix}.mp4'
        receipt_path=output.with_suffix('.encoding.json');intent_path=output.with_suffix('.encoding_intent.json')
        expected=dict(binding,variant=variant)
        if receipt_path.exists():
            done=read(receipt_path)
            if done.get('binding')!=expected or not output.exists()or sha(output)!=done['MP4_sha256']:
                raise ValueError('Existing movie belongs to another source/profile')
            probe(self.policy['ffprobe'],output,count);self.unchanged(binding);return done
        intent=read(intent_path)if intent_path.exists()else None
        if intent is not None and intent.get('binding')!=expected:
            raise ValueError('Existing encoder intent belongs to another source/profile')
        if output.exists():
            if not intent or intent.get('produced_sha256')!=sha(output):
                raise ValueError('Refuse overwriting foreign/unbound existing movie')
            detail=self.inspect(output,count,root/'frames',variant);self.unchanged(binding)
        else:
            if intent is None:
                nonce=str(uuid.uuid4());temp=output.with_name(output.stem+'.'+nonce+'.pending.mp4')
                intent=dict(binding=expected,nonce=nonce,temporary=str(temp),command=command(self.policy['ffmpeg'],root/'frames'/'%04d'/(variant+'.png'),count,temp))
                atomic(intent_path,intent);self.event('after_PREintent',intent)
            temp=Path(intent['temporary'])
            if temp.parent!=root or intent['nonce']not in temp.name:raise ValueError('Invalid owned temporary path')
            if temp.exists():
                try:detail=self.inspect(temp,count,root/'frames',variant)
                except(subprocess.SubprocessError,ValueError):
                    # Preserve an interrupted owned encode; never touch foreign files.
                    archive=temp.with_name(temp.name+'.interrupted.'+str(uuid.uuid4()))
                    temp.replace(archive);detail=None
            else:detail=None
            if detail is None:
                self.event('before_FFmpeg',intent)
                subprocess.run(intent['command'],check=True,pass_fds=(lock_fd,))
                self.event('after_FFmpeg',intent)
                detail=self.inspect(temp,count,root/'frames',variant)
            self.unchanged(binding);intent.update(produced_sha256=sha(temp),verification=detail)
            atomic(intent_path,intent);self.event('before_publish',intent)
            if output.exists():raise ValueError('Movie appeared during encode; refuse overwrite')
            temp.replace(output);self.event('after_publish',intent)
        done=dict(binding=expected,MP4_sha256=sha(output),verification=detail,PNG_masters_preserved=True,
            encoding_is_lossy=True,RGB_transfer_transform=False,adapter_sha256=sha(__file__))
        atomic(receipt_path,done);return done

    def __call__(self,spec,path,neural_only=False):
        self.guard();root=Path(spec['output']);count=72 if neural_only else 120
        lock=(root/'encoding.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();raise AwaitEncodingOwnership('An exact encoder owner still holds the inherited lock')
        try:
            binding=self.sources(spec,path,count);destination=root/('neural_3s_complete.json'if neural_only else'complete.json')
            if destination.exists()and read(destination).get('encoding_binding')!=binding:
                raise ValueError('Existing overall completion belongs to another encoder/source')
            artifacts={v:self.one(root,v,count,binding,lock.fileno())for v in ['lookcloser','hybrid']}
            self.unchanged(binding)
            done=dict(path_sha256=binding['path_sha256'],frame_count=count,fps=24,seconds=count/24,
                variants=['lookcloser','hybrid'],GT_last48_identical=not neural_only,visual_review_required=True,
                encoding_binding=binding,encoding_is_lossy=True,artifacts=artifacts,policy=self.policy)
            atomic(destination,done);return done
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def install(renderer,release_path):
    release_path=Path(release_path);original_release_sha=sha(release_path)
    def guard():
        release=read(release_path)
        if release.get('approved')is not True or sha(release_path)!=original_release_sha:
            raise ValueError('Encoding ROOT release missing or changed')
        for path,value in release['files_sha256'].items():
            if sha(path)!=value:raise ValueError('Encoding source/qualification changed '+path)
        resolved=set()
        for tool,item in release['tools'].items():
            if sha(item['path'])!=item['sha256']:raise ValueError('Encoding binary changed '+tool)
            actual=subprocess.check_output([item['path'],'-version'],text=True)
            if hashlib.sha256(actual.encode()).hexdigest()!=item['version_sha256']:
                raise ValueError('Encoding version/library ABI changed '+tool)
            linked=subprocess.check_output(['/usr/bin/ldd',item['path']],text=True)
            for line in linked.splitlines():
                match=re.search(r'=>\s+(/\S+)',line)or re.match(r'\s*(/\S+)\s+\(',line)
                if match:resolved.add(str(Path(match.group(1)).resolve()))
        if resolved!=set(release['resolved_library_sha256']):raise ValueError('Encoding dynamiclibrary selection changed')
        for path,value in release['resolved_library_sha256'].items():
            if sha(path)!=value:raise ValueError('Encoding resolved library changed '+path)
        return release
    release=guard();policy=dict(release['policy'],ROOT_encoding_release_sha256=original_release_sha);encoder=Encoder(policy,guard);renderer.encode=encoder;return encoder
