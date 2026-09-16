"""Run identical review using the executed consensus admission policy."""
import review_face_interior_visibility as review
from run_face_consensus_visibility import ROOT, consensus_proposals

if __name__ == '__main__':
    review.proposals = consensus_proposals
    review.__dict__['__file__'] = __file__
    review.torch.set_num_threads(2)
    with review.torch.inference_mode(): review.main(ROOT)
