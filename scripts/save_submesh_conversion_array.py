from tools.loading import *
from conf import *

def main():

    inds_cg1 = create_meshview_submesh_conversion_array(mesh_file_loc, "CG1")
    inds_cg2 = create_meshview_submesh_conversion_array(mesh_file_loc, "CG2")

    save_dir = "tools/conversion_arrays"

    np.save(save_dir+"/mv_to_sm_cg1.npy", inds_cg1)
    np.save(save_dir+"/mv_to_sm_cg2.npy", inds_cg2)

    return


if __name__ == "__main__":
    main()
