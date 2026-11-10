# Marshawn908–1268 selected native recipe

Primary: masked actor AlexLPIPS, without background.

Selected parameters:
```json
{
  "frequency_source_policy": "uniform16",
  "coarse_step": 0.000390625,
  "confidence_profile": "historical_power",
  "background_opacity_prior": 0.01,
  "model_units_scale": 0.2217203609534452,
  "gpu_cache": false
}
```

Steps: 150000. Native crops without resize; accepted V3 box.

|Arm|Confidence|Host|Step|Cap|LPIPS|PSNR|SSIM|
|---|---|---|---:|---|---:|---:|---:|
|uniform|linear/W64|ws07|8000|None|0.211019|29.4352|0.7074|
|uniform|linear/W64|ws07|8000|10|0.210975|29.4350|0.7074|
|uniform|linear/W64|ws07|16000|None|0.194231|29.6563|0.7089|
|uniform|linear/W64|ws07|16000|10|0.194135|29.6554|0.7088|
|uniform|linear/W64|ws07|32000|None|0.178308|29.6044|0.7066|
|uniform|linear/W64|ws07|32000|10|0.177967|29.6017|0.7064|
|uniform_repeat|linear/W64|silver0|8000|None|0.213298|29.4479|0.7076|
|uniform_repeat|linear/W64|silver0|8000|10|0.213258|29.4476|0.7075|
|uniform_repeat|linear/W64|silver0|16000|None|0.198578|29.6477|0.7093|
|uniform_repeat|linear/W64|silver0|16000|10|0.198475|29.6467|0.7092|
|uniform_repeat|linear/W64|silver0|32000|None|0.179406|29.6178|0.7065|
|uniform_repeat|linear/W64|silver0|32000|10|0.179095|29.6143|0.7063|
|uniform_gpu|linear/W64|silver1|8000|None|0.211276|29.5092|0.7074|
|uniform_gpu|linear/W64|silver1|8000|10|0.211232|29.5089|0.7074|
|uniform_gpu|linear/W64|silver1|16000|None|0.196134|29.7100|0.7089|
|uniform_gpu|linear/W64|silver1|16000|10|0.196026|29.7089|0.7088|
|uniform_gpu|linear/W64|silver1|32000|None|0.178218|29.6849|0.7064|
|uniform_gpu|linear/W64|silver1|32000|10|0.177880|29.6816|0.7062|
|learned095|linear/W64|silver2|8000|None|0.211140|29.4388|0.7053|
|learned095|linear/W64|silver2|8000|10|0.211097|29.4385|0.7053|
|learned095|linear/W64|silver2|16000|None|0.195093|29.4868|0.7047|
|learned095|linear/W64|silver2|16000|10|0.194973|29.4857|0.7046|
|learned095|linear/W64|silver2|32000|None|0.180215|29.5034|0.7030|
|learned095|linear/W64|silver2|32000|10|0.179872|29.4998|0.7027|
|learned090|linear/W64|silver3|8000|None|0.211239|29.4015|0.7069|
|learned090|linear/W64|silver3|8000|10|0.211197|29.4014|0.7069|
|learned090|linear/W64|silver3|16000|None|0.194986|29.5519|0.7066|
|learned090|linear/W64|silver3|16000|10|0.194888|29.5510|0.7065|
|learned090|linear/W64|silver3|32000|None|0.177422|29.5968|0.7042|
|learned090|linear/W64|silver3|32000|10|0.177093|29.5934|0.7039|
|learned090|linear/W64|silver3|48000|None|0.169158|29.5464|0.7019|
|learned090|linear/W64|silver3|48000|10|0.168479|29.5390|0.7013|
|learned090|linear/W64|silver3|64000|None|0.163628|29.5404|0.7002|
|learned090|linear/W64|silver3|64000|10|0.162687|29.5293|0.6993|
|learned090|linear/W64|silver3|75941|None|0.160875|29.5253|0.6994|
|learned090|linear/W64|silver3|75941|10|0.159660|29.5111|0.6982|
|learned090|linear/W64|silver3|80000|None|0.160269|29.5210|0.6988|
|learned090|linear/W64|silver3|80000|10|0.159152|29.5079|0.6977|
|learned090|linear/W64|silver3|96000|None|0.156920|29.4914|0.6976|
|learned090|linear/W64|silver3|96000|10|0.155552|29.4752|0.6963|
|arm_half|linear/W64|silver4|8000|None|0.205248|29.4799|0.7081|
|arm_half|linear/W64|silver4|8000|10|0.205213|29.4795|0.7081|
|arm_half|linear/W64|silver4|16000|None|0.186440|29.6201|0.7079|
|arm_half|linear/W64|silver4|16000|10|0.186309|29.6186|0.7078|
|arm_half|linear/W64|silver4|32000|None|0.167170|29.5412|0.7028|
|arm_half|linear/W64|silver4|32000|10|0.166759|29.5370|0.7024|
|arm_half|linear/W64|silver4|48000|None|0.158786|29.5376|0.7001|
|arm_half|linear/W64|silver4|48000|10|0.158011|29.5292|0.6994|
|arm_half|linear/W64|silver4|64000|None|0.154139|29.4767|0.6976|
|arm_half|linear/W64|silver4|64000|10|0.152993|29.4641|0.6966|
|arm_half|linear/W64|silver4|75941|None|0.149795|29.4555|0.6954|
|arm_half|linear/W64|silver4|75941|10|0.148383|29.4391|0.6942|
|arm_half|linear/W64|silver4|80000|None|0.149533|29.4411|0.6952|
|arm_half|linear/W64|silver4|80000|10|0.148266|29.4263|0.6941|
|arm_half|linear/W64|silver4|96000|None|0.146744|29.3992|0.6934|
|arm_half|linear/W64|silver4|96000|10|0.145254|29.3814|0.6920|
|arm_half|linear/W64|silver4|112000|None|0.142419|29.3860|0.6910|
|arm_half|linear/W64|silver4|112000|10|0.140497|29.3613|0.6891|
|arm_half|linear/W64|silver4|128000|None|0.140410|29.3477|0.6896|
|arm_half|linear/W64|silver4|128000|10|0.138296|29.3198|0.6875|
|arm_half|linear/W64|silver4|144000|None|0.139611|29.3226|0.6890|
|arm_half|linear/W64|silver4|144000|10|0.137385|29.2930|0.6868|
|arm_half|linear/W64|silver4|150000|None|0.139479|29.3250|0.6887|
|arm_half|linear/W64|silver4|150000|10|0.137206|29.2949|0.6864|
|arm_double|linear/W64|silver5|8000|None|0.222303|29.3500|0.7040|
|arm_double|linear/W64|silver5|8000|10|0.222276|29.3498|0.7040|
|arm_double|linear/W64|silver5|16000|None|0.210345|29.5318|0.7076|
|arm_double|linear/W64|silver5|16000|10|0.210267|29.5311|0.7075|
|arm_double|linear/W64|silver5|32000|None|0.191630|29.6492|0.7082|
|arm_double|linear/W64|silver5|32000|10|0.191442|29.6467|0.7080|
|arm_2048|linear/W64|silver6|8000|None|0.212510|29.4299|0.7072|
|arm_2048|linear/W64|silver6|8000|10|0.212469|29.4295|0.7072|
|arm_2048|linear/W64|silver6|16000|None|0.194132|29.6101|0.7086|
|arm_2048|linear/W64|silver6|16000|10|0.194020|29.6093|0.7085|
|arm_2048|linear/W64|silver6|32000|None|0.177763|29.5696|0.7059|
|arm_2048|linear/W64|silver6|32000|10|0.177455|29.5668|0.7057|
|arm_2048|linear/W64|silver6|48000|None|0.167871|29.5394|0.7035|
|arm_2048|linear/W64|silver6|48000|10|0.167224|29.5326|0.7030|
|arm_2048|linear/W64|silver6|64000|None|0.161199|29.5367|0.7018|
|arm_2048|linear/W64|silver6|64000|10|0.160131|29.5257|0.7009|
|arm_2048|linear/W64|silver6|75941|None|0.157117|29.4872|0.6999|
|arm_2048|linear/W64|silver6|75941|10|0.155789|29.4727|0.6988|
|arm_2048|linear/W64|silver6|80000|None|0.156869|29.4729|0.6994|
|arm_2048|linear/W64|silver6|80000|10|0.155607|29.4592|0.6984|
|arm_max3|linear/W64|ws08|8000|None|0.213191|29.4826|0.7080|
|arm_max3|linear/W64|ws08|8000|10|0.213137|29.4822|0.7080|
|arm_max3|linear/W64|ws08|16000|None|0.195930|29.6489|0.7090|
|arm_max3|linear/W64|ws08|16000|10|0.195795|29.6479|0.7090|
|arm_max3|linear/W64|ws08|32000|None|0.178778|29.6079|0.7067|
|arm_max3|linear/W64|ws08|32000|10|0.178339|29.6039|0.7064|
|fr_initial03|linear/W64|silver7|8000|None|0.214531|29.3710|0.7078|
|fr_initial03|linear/W64|silver7|8000|10|0.214514|29.3708|0.7078|
|fr_initial03|linear/W64|silver7|16000|None|0.199457|29.3882|0.7081|
|fr_initial03|linear/W64|silver7|16000|10|0.199386|29.3878|0.7080|
|fr_initial03|linear/W64|silver7|32000|None|0.181795|29.3808|0.7056|
|fr_initial03|linear/W64|silver7|32000|10|0.181555|29.3788|0.7054|
|fr_initial03|linear/W64|silver7|48000|None|0.171356|29.3478|0.7035|
|fr_initial03|linear/W64|silver7|48000|10|0.170847|29.3431|0.7031|
|fr_initial03|linear/W64|silver7|64000|None|0.165507|29.3240|0.7016|
|fr_initial03|linear/W64|silver7|64000|10|0.164702|29.3159|0.7010|
|fr_initial03|linear/W64|silver7|75941|None|0.161886|29.2888|0.7002|
|fr_initial03|linear/W64|silver7|75941|10|0.160862|29.2780|0.6993|
|fr_initial03|linear/W64|silver7|80000|None|0.160996|29.3000|0.6998|
|fr_initial03|linear/W64|silver7|80000|10|0.159874|29.2877|0.6988|
|resolution16384|linear/W64|clever|8000|None|0.214906|29.4133|0.7071|
|resolution16384|linear/W64|clever|8000|10|0.214894|29.4132|0.7071|
|resolution16384|linear/W64|clever|16000|None|0.200798|29.5643|0.7083|
|resolution16384|linear/W64|clever|16000|10|0.200764|29.5640|0.7083|
|resolution16384|linear/W64|clever|32000|None|0.182513|29.4985|0.7062|
|resolution16384|linear/W64|clever|32000|10|0.182376|29.4975|0.7061|
|frequency512|linear/W64|olive|8000|None|0.213245|29.3575|0.7075|
|frequency512|linear/W64|olive|8000|10|0.213186|29.3570|0.7075|
|frequency512|linear/W64|olive|16000|None|0.192433|29.5247|0.7083|
|frequency512|linear/W64|olive|16000|10|0.192225|29.5232|0.7081|
|frequency512|linear/W64|olive|32000|None|0.172693|29.5547|0.7052|
|frequency512|linear/W64|olive|32000|10|0.172010|29.5478|0.7047|
|batch16384|linear/W64|ws08|8000|None|0.198960|29.4240|0.7055|
|batch16384|linear/W64|ws08|8000|10|0.198921|29.4237|0.7055|
|batch16384|linear/W64|ws08|16000|None|0.183743|29.4760|0.7045|
|batch16384|linear/W64|ws08|16000|10|0.183610|29.4746|0.7044|
|batch16384|linear/W64|ws08|32000|None|0.166858|29.4224|0.6995|
|batch16384|linear/W64|ws08|32000|10|0.166395|29.4170|0.6991|
|batch16384|linear/W64|ws08|48000|None|0.157292|29.3251|0.6959|
|batch16384|linear/W64|ws08|48000|10|0.156352|29.3143|0.6950|
|batch16384|linear/W64|ws08|64000|None|0.150453|29.2699|0.6929|
|batch16384|linear/W64|ws08|64000|10|0.149000|29.2529|0.6916|
|batch16384|linear/W64|ws08|75941|None|0.147135|29.2361|0.6909|
|batch16384|linear/W64|ws08|75941|10|0.145357|29.2144|0.6892|
|batch16384|linear/W64|ws08|80000|None|0.147115|29.2237|0.6905|
|batch16384|linear/W64|ws08|80000|10|0.145464|29.2040|0.6889|
|batch16384|linear/W64|ws08|96000|None|0.144080|29.1829|0.6888|
|batch16384|linear/W64|ws08|96000|10|0.142047|29.1582|0.6869|
|batch16384|linear/W64|ws08|112000|None|0.141977|29.1617|0.6877|
|batch16384|linear/W64|ws08|112000|10|0.139684|29.1325|0.6854|
|batch16384|linear/W64|ws08|128000|None|0.140487|29.1296|0.6868|
|batch16384|linear/W64|ws08|128000|10|0.137964|29.0970|0.6843|
|head20|linear/W64|ws07|8000|None|0.213579|29.4535|0.7073|
|head20|linear/W64|ws07|8000|10|0.213542|29.4532|0.7073|
|head20|linear/W64|ws07|16000|None|0.195204|29.5398|0.7075|
|head20|linear/W64|ws07|16000|10|0.195112|29.5390|0.7074|
|head20|linear/W64|ws07|32000|None|0.178743|29.6715|0.7058|
|head20|linear/W64|ws07|32000|10|0.178406|29.6683|0.7055|
|rig_units|linear/W64|silver5|8000|None|0.199861|29.4254|0.7081|
|rig_units|linear/W64|silver5|8000|10|0.199784|29.4246|0.7080|
|rig_units|linear/W64|silver5|16000|None|0.178557|29.4745|0.7053|
|rig_units|linear/W64|silver5|16000|10|0.178283|29.4704|0.7049|
|rig_units|linear/W64|silver5|32000|None|0.160506|29.3639|0.6988|
|rig_units|linear/W64|silver5|32000|10|0.159664|29.3526|0.6979|
|rig_units|linear/W64|silver5|48000|None|0.151549|29.2278|0.6944|
|rig_units|linear/W64|silver5|48000|10|0.150246|29.2095|0.6929|
|rig_units|linear/W64|silver5|64000|None|0.146752|29.2397|0.6914|
|rig_units|linear/W64|silver5|64000|10|0.145019|29.2149|0.6894|
|rig_units|linear/W64|silver5|75941|None|0.143953|29.1958|0.6894|
|rig_units|linear/W64|silver5|75941|10|0.142002|29.1667|0.6870|
|rig_units|linear/W64|silver5|80000|None|0.143327|29.1781|0.6888|
|rig_units|linear/W64|silver5|80000|10|0.141472|29.1508|0.6866|
|rig_units|linear/W64|silver5|96000|None|0.141705|29.1636|0.6874|
|rig_units|linear/W64|silver5|96000|10|0.139619|29.1319|0.6848|
|confidence32|linear/W32|silver0|8000|None|0.211695|29.5261|0.7086|
|confidence32|linear/W32|silver0|8000|10|0.211652|29.5258|0.7085|
|confidence32|linear/W32|silver0|16000|None|0.196073|29.6272|0.7098|
|confidence32|linear/W32|silver0|16000|10|0.195952|29.6263|0.7098|
|confidence32|linear/W32|silver0|32000|None|0.178036|29.6452|0.7072|
|confidence32|linear/W32|silver0|32000|10|0.177691|29.6418|0.7070|
|learned095_arm_half|linear/W64|silver1|8000|None|0.202095|29.4573|0.7063|
|learned095_arm_half|linear/W64|silver1|8000|10|0.202055|29.4568|0.7062|
|learned095_arm_half|linear/W64|silver1|16000|None|0.183304|29.4918|0.7039|
|learned095_arm_half|linear/W64|silver1|16000|10|0.183156|29.4903|0.7037|
|learned095_arm_half|linear/W64|silver1|32000|None|0.168613|29.4890|0.7005|
|learned095_arm_half|linear/W64|silver1|32000|10|0.168153|29.4841|0.7001|
|uniform_clever|linear/W64|silver2|8000|None|0.213725|29.4909|0.7079|
|uniform_clever|linear/W64|silver2|8000|10|0.213681|29.4905|0.7078|
|uniform_clever|linear/W64|silver2|16000|None|0.197188|29.7069|0.7094|
|uniform_clever|linear/W64|silver2|16000|10|0.197097|29.7060|0.7093|
|uniform_clever|linear/W64|silver2|32000|None|0.179689|29.7246|0.7071|
|uniform_clever|linear/W64|silver2|32000|10|0.179367|29.7215|0.7069|
|clear_background|linear/W64|olive|8000|None|0.206743|29.5400|0.7088|
|clear_background|linear/W64|olive|8000|10|0.206700|29.5397|0.7087|
|clear_background|linear/W64|olive|16000|None|0.186029|29.6798|0.7080|
|clear_background|linear/W64|olive|16000|10|0.185903|29.6784|0.7079|
|clear_background|linear/W64|olive|32000|None|0.168201|29.6639|0.7040|
|clear_background|linear/W64|olive|32000|10|0.167778|29.6594|0.7036|
|clear_background|linear/W64|olive|48000|None|0.158110|29.6387|0.7006|
|clear_background|linear/W64|olive|48000|10|0.157262|29.6297|0.6999|
|clear_background|linear/W64|olive|64000|None|0.153500|29.6118|0.6983|
|clear_background|linear/W64|olive|64000|10|0.152359|29.5987|0.6974|
|clear_background|linear/W64|olive|75941|None|0.149538|29.5634|0.6964|
|clear_background|linear/W64|olive|75941|10|0.148127|29.5470|0.6952|
|clear_background|linear/W64|olive|80000|None|0.149802|29.5497|0.6961|
|clear_background|linear/W64|olive|80000|10|0.148525|29.5349|0.6950|
|clear_background|linear/W64|olive|96000|None|0.146295|29.5213|0.6941|
|clear_background|linear/W64|olive|96000|10|0.144733|29.5026|0.6928|
|clear_background|linear/W64|olive|112000|None|0.142438|29.4957|0.6920|
|clear_background|linear/W64|olive|112000|10|0.140452|29.4704|0.6901|
|arm_quarter|linear/W64|ws07|8000|None|0.197457|29.5523|0.7088|
|arm_quarter|linear/W64|ws07|8000|10|0.197400|29.5519|0.7087|
|arm_quarter|linear/W64|ws07|16000|None|0.176870|29.6409|0.7069|
|arm_quarter|linear/W64|ws07|16000|10|0.176665|29.6387|0.7067|
|arm_quarter|linear/W64|ws07|32000|None|0.160298|29.5851|0.7018|
|arm_quarter|linear/W64|ws07|32000|10|0.159693|29.5784|0.7012|
|arm_quarter|linear/W64|ws07|48000|None|0.151042|29.5025|0.6974|
|arm_quarter|linear/W64|ws07|48000|10|0.150037|29.4905|0.6965|
|arm_quarter|linear/W64|ws07|64000|None|0.146713|29.4779|0.6949|
|arm_quarter|linear/W64|ws07|64000|10|0.145348|29.4615|0.6937|
|arm_quarter|linear/W64|ws07|75941|None|0.142404|29.4198|0.6923|
|arm_quarter|linear/W64|ws07|75941|10|0.140807|29.3998|0.6909|
|arm_quarter|linear/W64|ws07|80000|None|0.142838|29.4066|0.6922|
|arm_quarter|linear/W64|ws07|80000|10|0.141343|29.3882|0.6908|
|arm_quarter|linear/W64|ws07|96000|None|0.140472|29.3861|0.6908|
|arm_quarter|linear/W64|ws07|96000|10|0.138816|29.3649|0.6892|
|arm_quarter|linear/W64|ws07|112000|None|0.136421|29.3656|0.6883|
|arm_quarter|linear/W64|ws07|112000|10|0.134461|29.3384|0.6863|
|arm_quarter|linear/W64|ws07|128000|None|0.134492|29.3302|0.6866|
|arm_quarter|linear/W64|ws07|128000|10|0.132385|29.3003|0.6844|
|arm_quarter|linear/W64|ws07|144000|None|0.134280|29.3066|0.6861|
|arm_quarter|linear/W64|ws07|144000|10|0.132092|29.2753|0.6838|
|arm_quarter|linear/W64|ws07|150000|None|0.134129|29.3143|0.6860|
|arm_quarter|linear/W64|ws07|150000|10|0.131931|29.2824|0.6837|
|exr|linear/W64|ws08|8000|None|0.209742|29.4778|0.7092|
|exr|linear/W64|ws08|8000|10|0.209697|29.4775|0.7091|
|exr|linear/W64|ws08|16000|None|0.186628|29.6744|0.7088|
|exr|linear/W64|ws08|16000|10|0.186456|29.6727|0.7087|
|exr|linear/W64|ws08|32000|None|0.168317|29.6033|0.7042|
|exr|linear/W64|ws08|32000|10|0.167769|29.5976|0.7037|
|reference_half|historical_power/W64|silver0|8000|None|0.204257|29.5993|0.7094|
|reference_half|historical_power/W64|silver0|8000|10|0.204215|29.5988|0.7093|
|reference_half|historical_power/W64|silver0|16000|None|0.184644|29.6137|0.7080|
|reference_half|historical_power/W64|silver0|16000|10|0.184487|29.6123|0.7078|
|reference_half|historical_power/W64|silver0|32000|None|0.166579|29.6134|0.7033|
|reference_half|historical_power/W64|silver0|32000|10|0.166072|29.6079|0.7028|
|reference_half|historical_power/W64|silver0|48000|None|0.155371|29.5039|0.6990|
|reference_half|historical_power/W64|silver0|48000|10|0.154399|29.4932|0.6981|
|reference_half|historical_power/W64|silver0|64000|None|0.150353|29.5138|0.6969|
|reference_half|historical_power/W64|silver0|64000|10|0.149010|29.4974|0.6956|
|reference_half|historical_power/W64|silver0|75941|None|0.146816|29.4325|0.6945|
|reference_half|historical_power/W64|silver0|75941|10|0.145211|29.4123|0.6929|
|reference_half|historical_power/W64|silver0|80000|None|0.146316|29.4252|0.6943|
|reference_half|historical_power/W64|silver0|80000|10|0.144791|29.4065|0.6928|
|reference_half|historical_power/W64|silver0|96000|None|0.143571|29.4064|0.6926|
|reference_half|historical_power/W64|silver0|96000|10|0.141796|29.3839|0.6908|
|reference_half|historical_power/W64|silver0|112000|None|0.138904|29.3779|0.6898|
|reference_half|historical_power/W64|silver0|112000|10|0.136732|29.3472|0.6873|
|reference_uniform|historical_power/W64|silver2|8000|None|0.209852|29.4680|0.7083|
|reference_uniform|historical_power/W64|silver2|8000|10|0.209801|29.4677|0.7083|
|reference_uniform|historical_power/W64|silver2|16000|None|0.195941|29.5888|0.7085|
|reference_uniform|historical_power/W64|silver2|16000|10|0.195839|29.5876|0.7084|
|reference_uniform|historical_power/W64|silver2|32000|None|0.178927|29.4331|0.7052|
|reference_uniform|historical_power/W64|silver2|32000|10|0.178566|29.4287|0.7048|
|reference_learned_half|historical_power/W64|silver1|8000|None|0.200648|29.6063|0.7079|
|reference_learned_half|historical_power/W64|silver1|8000|10|0.200588|29.6059|0.7078|
|reference_learned_half|historical_power/W64|silver1|16000|None|0.181993|29.5440|0.7048|
|reference_learned_half|historical_power/W64|silver1|16000|10|0.181821|29.5421|0.7046|
|reference_learned_half|historical_power/W64|silver1|32000|None|0.167299|29.6185|0.7020|
|reference_learned_half|historical_power/W64|silver1|32000|10|0.166771|29.6130|0.7016|
|reference_learned_half|historical_power/W64|silver1|48000|None|0.159032|29.5373|0.6988|
|reference_learned_half|historical_power/W64|silver1|48000|10|0.158085|29.5270|0.6979|
|reference_learned_half|historical_power/W64|silver1|64000|None|0.154021|29.5172|0.6966|
|reference_learned_half|historical_power/W64|silver1|64000|10|0.152682|29.5023|0.6953|
|reference_learned_half|historical_power/W64|silver1|75941|None|0.151487|29.4800|0.6952|
|reference_learned_half|historical_power/W64|silver1|75941|10|0.149945|29.4622|0.6937|
|reference_learned_half|historical_power/W64|silver1|80000|None|0.151432|29.4790|0.6949|
|reference_learned_half|historical_power/W64|silver1|80000|10|0.149963|29.4626|0.6936|
|reference_learned_half|historical_power/W64|silver1|96000|None|0.146592|29.4419|0.6921|
|reference_learned_half|historical_power/W64|silver1|96000|10|0.144706|29.4185|0.6902|
|reference_learned_half|historical_power/W64|silver1|112000|None|0.143559|29.4150|0.6903|
|reference_learned_half|historical_power/W64|silver1|112000|10|0.141368|29.3869|0.6881|
|reference_background|historical_power/W64|silver3|8000|None|0.204730|29.6982|0.7098|
|reference_background|historical_power/W64|silver3|8000|10|0.204693|29.6977|0.7098|
|reference_background|historical_power/W64|silver3|16000|None|0.181720|29.8031|0.7085|
|reference_background|historical_power/W64|silver3|16000|10|0.181560|29.8013|0.7083|
|reference_background|historical_power/W64|silver3|32000|None|0.164832|29.7748|0.7037|
|reference_background|historical_power/W64|silver3|32000|10|0.164264|29.7685|0.7032|
|reference_background|historical_power/W64|silver3|48000|None|0.155043|29.7438|0.6998|
|reference_background|historical_power/W64|silver3|48000|10|0.153999|29.7316|0.6989|
|reference_background|historical_power/W64|silver3|64000|None|0.149657|29.6897|0.6974|
|reference_background|historical_power/W64|silver3|64000|10|0.148243|29.6722|0.6961|
|reference_background|historical_power/W64|silver3|75941|None|0.146671|29.6632|0.6958|
|reference_background|historical_power/W64|silver3|75941|10|0.144996|29.6415|0.6942|
|reference_background|historical_power/W64|silver3|80000|None|0.146261|29.6631|0.6954|
|reference_background|historical_power/W64|silver3|80000|10|0.144699|29.6431|0.6939|
|reference_background|historical_power/W64|silver3|96000|None|0.143589|29.6244|0.6939|
|reference_background|historical_power/W64|silver3|96000|10|0.141753|29.6009|0.6922|
|reference_background|historical_power/W64|silver3|112000|None|0.138817|29.5951|0.6909|
|reference_background|historical_power/W64|silver3|112000|10|0.136499|29.5623|0.6885|
|power_quarter|historical_power/W64|silver6|8000|None|0.197553|29.7123|0.7096|
|power_quarter|historical_power/W64|silver6|8000|10|0.197491|29.7118|0.7095|
|power_quarter|historical_power/W64|silver6|16000|None|0.176093|29.6792|0.7066|
|power_quarter|historical_power/W64|silver6|16000|10|0.175865|29.6766|0.7064|
|power_quarter|historical_power/W64|silver6|32000|None|0.158440|29.5774|0.7007|
|power_quarter|historical_power/W64|silver6|32000|10|0.157821|29.5698|0.7001|
|power_quarter|historical_power/W64|silver6|48000|None|0.150018|29.5340|0.6964|
|power_quarter|historical_power/W64|silver6|48000|10|0.149039|29.5214|0.6954|
|power_quarter|historical_power/W64|silver6|64000|None|0.144080|29.5098|0.6937|
|power_quarter|historical_power/W64|silver6|64000|10|0.142729|29.4924|0.6924|
|power_quarter|historical_power/W64|silver6|75941|None|0.141313|29.4620|0.6918|
|power_quarter|historical_power/W64|silver6|75941|10|0.139820|29.4414|0.6902|
|power_quarter|historical_power/W64|silver6|80000|None|0.141215|29.4495|0.6910|
|power_quarter|historical_power/W64|silver6|80000|10|0.139768|29.4306|0.6896|
|power_quarter_background|historical_power/W64|silver7|8000|None|0.198550|29.7183|0.7101|
|power_quarter_background|historical_power/W64|silver7|8000|10|0.198520|29.7179|0.7100|
|power_quarter_background|historical_power/W64|silver7|16000|None|0.176771|29.7748|0.7078|
|power_quarter_background|historical_power/W64|silver7|16000|10|0.176570|29.7721|0.7076|
|power_quarter_background|historical_power/W64|silver7|32000|None|0.159103|29.7174|0.7021|
|power_quarter_background|historical_power/W64|silver7|32000|10|0.158535|29.7098|0.7015|
|power_quarter_background|historical_power/W64|silver7|48000|None|0.149799|29.6431|0.6977|
|power_quarter_background|historical_power/W64|silver7|48000|10|0.148806|29.6298|0.6967|
|power_quarter_background|historical_power/W64|silver7|64000|None|0.144791|29.5863|0.6946|
|power_quarter_background|historical_power/W64|silver7|64000|10|0.143514|29.5682|0.6932|
|power_quarter_background|historical_power/W64|silver7|75941|None|0.141627|29.5573|0.6925|
|power_quarter_background|historical_power/W64|silver7|75941|10|0.140148|29.5351|0.6909|
|power_quarter_background|historical_power/W64|silver7|80000|None|0.140898|29.5627|0.6923|
|power_quarter_background|historical_power/W64|silver7|80000|10|0.139518|29.5425|0.6908|
|reference_background_strong|historical_power/W64|silver2|8000|None|0.204494|29.6686|0.7099|
|reference_background_strong|historical_power/W64|silver2|8000|10|0.204464|29.6682|0.7099|
|reference_background_strong|historical_power/W64|silver2|16000|None|0.183268|29.7505|0.7089|
|reference_background_strong|historical_power/W64|silver2|16000|10|0.183102|29.7489|0.7087|
|reference_background_strong|historical_power/W64|silver2|32000|None|0.167648|29.6036|0.7045|
|reference_background_strong|historical_power/W64|silver2|32000|10|0.167160|29.5984|0.7040|
|reference_background_strong|historical_power/W64|silver2|48000|None|0.155550|29.4706|0.7002|
|reference_background_strong|historical_power/W64|silver2|48000|10|0.154578|29.4602|0.6993|
|reference_background_strong|historical_power/W64|silver2|64000|None|0.150391|29.4507|0.6977|
|reference_background_strong|historical_power/W64|silver2|64000|10|0.149041|29.4353|0.6965|
|reference_background_strong|historical_power/W64|silver2|75941|None|0.146434|29.4490|0.6955|
|reference_background_strong|historical_power/W64|silver2|75941|10|0.144728|29.4291|0.6939|
|reference_background_strong|historical_power/W64|silver2|80000|None|0.146348|29.4388|0.6954|
|reference_background_strong|historical_power/W64|silver2|80000|10|0.144761|29.4203|0.6939|

All final models cold-start on72TRAIN cameras; metrics on former heldout cameras are in-sample. Maximum150000attempts including recovery replay. FPS120 remains assumed. Plateau is only asserted if the saved late trajectory passes the two-signal test. EXR sources are decoded from PQ16, not independent higher-information originals.

Confidence-port correction: early runs used a linear outside taper. The historical recipe uses exponent log(0.1)/log(0.5), giving weight0.1 at half width. Reference arms recompute the sealed helper on qualified actors, also removing confidence halos from rejected crew. Early linear runs remain explicitly labelled alternative measurements, not exact historical-recipe baselines.

Inference cap: 10. Plateau confirmed: False. Trajectory completed iterations: 88476.

Winning validation checkpoint: 80000steps. Final training policy: Plateau not confirmed; allow the150k attempt ceiling and retain the best scored per-frame checkpoint; winning validation step is reported separately, not relabelled as a150k measurement.

Hyperparameters compared at 80000 common steps across 11 continued finalists. The winning arm/cap then uses its own best measured checkpoint; unequal run lengths do not decide the hyperparameter comparison. Full comparison membership and evidence: selected_recipe.json.

Effective training settings:
```json
{
  "hash_levels": 16,
  "hash_features_per_level": 2,
  "log2_hashmap_size": 25,
  "max_res": 8192,
  "sh_degree": 4,
  "correct_sh_directions": true,
  "density_activation": "trunc_exp",
  "train_rays_per_batch": 8192,
  "sampling": "body80 with exact rounded proposal expectation, no 1/q correction",
  "head_fraction": 0.15,
  "confidence_width_native": 64,
  "opaque_core_erosion_native_pixels": 32,
  "opaque_core_prior": 0.01,
  "adam_betas": [
    0.9,
    0.99
  ],
  "learning_rate": 0.01,
  "learning_rate_final": 0.0001,
  "learning_rate_horizon": 200000,
  "warmup_steps": 4096,
  "distortion_weight_start": 0.01,
  "distortion_weight_final": 0.1,
  "distortion_ramp": [
    4096,
    12288
  ],
  "fr_initial": 1.0,
  "fr_switch_step": 75941,
  "fr_after_switch": 0.3,
  "coarse_step": 0.000390625,
  "minimum_step": 0.0001,
  "maximum_step": 0.1,
  "max_samples_per_ray": 1024,
  "corrected_arm_allocator": true,
  "interval_level_mode": "midpoint",
  "occupancy_resolution": 128,
  "occupancy_ema": 0.95,
  "occupancy_threshold": 0.01,
  "occupancy_update_interval": 16,
  "occupancy_dilation_radius": 0,
  "frequency_grid_update_interval": 1024,
  "frequency_projection": "legacy",
  "frequency_probe_train_rays": 2048,
  "frequency_source_policy": "uniform16",
  "native_rgb_resize": false,
  "seed": 42,
  "background_opacity_prior": 0.01,
  "confidence_profile": "historical_power",
  "confidence_exponent": 3.321928094887362
}
```

Fixed tuning heldouts: CB, GC, NB. Canonical physical rig geometry (apply world_to_box before testing aabb; then apply the isotropic model-unit scale above):
```json
{
  "aabb": [
    [
      -0.41,
      -1.2,
      -0.15
    ],
    [
      3.41,
      5.14,
      1.77
    ]
  ],
  "world_to_box": [
    [
      0.6478439998448969,
      -0.31083015727912405,
      0.6954730513763996,
      0.0
    ],
    [
      -0.7600046002119942,
      -0.201556873214209,
      0.6178736396033724,
      0.0
    ],
    [
      -0.05187638693636899,
      -0.9288484484489095,
      -0.36680976035607593,
      1.8163470591461168
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  ],
  "box_to_world": [
    [
      0.6478439998448972,
      -0.760004600211994,
      -0.051876386936369014,
      0.0942255228509999
    ],
    [
      -0.31083015727912416,
      -0.20155687321420906,
      -0.9288484484489095,
      1.6871111477326102
    ],
    [
      0.6954730513763997,
      0.6178736396033724,
      -0.36680976035607593,
      0.6662538294888504
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  ]
}
```

Rig calibration: 1.0805units/metre. Physical box dimensions: 3.5354m, 5.8677m, 1.7770m.

GPU image cache benchmark:
```json
{
  "adopted": false,
  "minimum_speedup": 0.05,
  "reason": "Matched cold native runs on Silver; includes checkpoint/evaluation overhead",
  "speedup": 0.0243041548562708,
  "seconds_per_step": {
    "uniform_repeat": 0.10557378505170345,
    "uniform_gpu": 0.10306878533214331
  }
}
```

EXR detail adoption checks passed: 0 / 6. Full head/view evidence and fixed-reference policy are recorded in selected_recipe.json.

Additional diagnostics and review: [runtime and mask checks](marshawn_908_1268_diagnostics/), [matched80k raw FHD comparison on Drive](https://drive.google.com/file/d/10Ormx7gGEV4bd397QNt7RpKuQzqo31DX/view). Raw unmasked renders retain contour halos and some unsupported opacity. The selected recipe wins the declared actor metric; a clean foreground matte is not established by this experiment. Native crops remain at source resolution. PNG uses the fixed PQ/AP1-D65 decode and1.6/Reinhard/sRGB display path in `prepare.py`; EXR is derived from the same PQ16 originals.

Production allocation after the formerly busy Clever GPU became available: WS07/WS08 41frames each, Silver8GPUs29frames each, Olive23frames, Clever24frames;361unique frames total. Only never-started tail jobs were moved; existing training/preparation parents were preserved. All72cameras participate in every final model. Services continue independently of the interactive session; completed models are tensor-verified before optimizer checkpoints are released. The machine-readable frozen recipe is [marshawn_908_1268_selected_recipe.json](marshawn_908_1268_selected_recipe.json).

First72-camera production spec passed parameter, source/extension SHA, head/admission coverage and native-resolution checks: [audit](marshawn_908_1268_diagnostics/first_production_spec_audit.json). Clever recovery passed an actual256-ray CUDA forward/backward and full native-image SAM forward before enrollment: [qualification](marshawn_908_1268_diagnostics/clever_production_gpu_requalification.json), [allocation receipt](marshawn_908_1268_diagnostics/clever_production_enrollment.json). Raw winner80k contact sheet, visually reviewed across CB/GC/NB and nearby novel views: [Drive](https://drive.google.com/file/d/1Q3o86ptWya4MoLjYWmENt-XT-1GIcJPS/view). All four performers remain visible, subject to source-FOV occlusion/cutoff; contour halos persist.

Preparation overlap: after first-frame native transfers and72-camera spec verification, the production watcher was extended to request one future frame per remote GPU queue on its source owner. This runs on CPU during current SGD, with the existing source-frame lease acquired nonblocking. The foreground production path still owns transfer verification/release and SAM; numerical training code/recipe are unchanged. Queue-selection and actual conflicting-lease checks passed. No measured end-to-end speedup is claimed yet.

Frame908 completed150000 iterations; selected144000 cap10: actor LPIPS0.108058, PSNR32.53593dB, SSIM0.799172 (all72TRAIN, in-sample). Independent streamed export SHA, recipe/AABB hashes and tensor-roundtrip receipt passed: [audit](marshawn_908_1268_diagnostics/frame908_completed_model_audit.json). WS07 advanced automatically to913. Full-FOV Full HD and nearby novel-view renders from the selected export were visually inspected across CB/GC/NB: all four performers remain represented, with source-FOV cutoffs/occlusions; substantial contour halos and unsupported opacity remain. This is not a clean-matte sign-off. [Six-view raw contact on Drive](https://drive.google.com/file/d/1tEodzaUtMJJ6EiEuZsyJDJFy-qdRBes_/view). Both render receipts match the independently verified model SHA and1920x1080 canvas.

Independent final export auditing is available in `scripts/marshawn_sequence/audit_exports.py`. It streams model SHA on the artifact host, checks recipe/AABB,72TRAIN/native preparation, selected parameters, cold-start receipts, iteration/attempt ceiling, selected best committed actor LPIPS, and export tensor-roundtrip receipts. It does not independently reload/recompare the deleted optimizer tensors or establish visual quality. Use `--root ROOT --queue QUEUE --output REPORT --require-complete`; for archived Clever frames use `--frames-dir ARCHIVE` on Silver and a copy of Clever’s queue. Silver needs one queue report per GPU. Combine all host/GPU reports with `--combine REPORT... --output GLOBAL --require-complete`: the global gate requires exactly908–1268 with no duplicate IDs/model digests. Missing reports remain pending. Actual WS07/WS08 exports passed (one each); partial completion returned nonzero under `--require-complete`, and repeated queue coverage was rejected. The combined partial audit reports2verified/359pending, not campaign completion. Training services were unchanged.

The export audit additionally checks actual `transforms.json`:72unique physical cameras, exactly all72image filenames in train, transforms SHA, and each training width/height equal to its source native crop rectangle. Both completed production models passed these stronger checks.

Frame914128k visual follow-up: [GC sampled-actor overview](marshawn_908_1268_diagnostics/frame914_GC_128000_review.jpg), [native head GT/prediction](marshawn_908_1268_diagnostics/frame914_GC_head1_128000.png), [source subject-alpha comparison](marshawn_908_1268_diagnostics/frame914_GC_head1_source_mask_review.jpg). Head/hair remains covered but reconstructed hair is smoother; contour halos persist. The inspected source semantic `soft_alpha` is uint8 (0–255), has speckled/nonuniform interior and stair-step edge regions. No gross head cutoff was found in this patch; it does not establish all masks are correct or attribute blur solely to masks. `native/masks/*.png` is the all-valid sampling canvas, not the subject alpha. Overview rectangle edges are unqueried metric ROI boundaries, not evidence of AABB clipping. Frozen source masks/recipe remain unchanged.

Frame913 initial8k cap10 visual gate completed: [GC overview](marshawn_908_1268_diagnostics/frame913_GC_8000_review.jpg), [native head GT/pred](marshawn_908_1268_diagnostics/frame913_GC_head1_8000.png). In-sample actor metrics(all72TRAIN): PSNR30.934797dB, SSIM0.739173, LPIPS0.198042. Pose/face structure is present; hair remains smoother and contour halos persist. ROI exterior is unqueried; this overview is not a full-FOV/AABB clipping test. Continue frozen recipe toward150k.
