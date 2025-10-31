import rp
rp.git_import('ControlNet')

from rp.git.ControlNet.annotator import (
    run_hed,
    run_midas,
    run_midas_normals,
    run_openpose,
    run_uniformer,
    run_annotator_demo,
)

if __name__=='__main__':
    run_annotator_demo()
