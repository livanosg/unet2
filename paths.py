from glob import glob
from config import paths


def get_paths(dataset, modality):
    if modality in ('CT', 'ALL'):
        ct_dcm_paths = paths[dataset] + '/**/CT/**/**.dcm'
        ct_grd_paths = paths[dataset] + '/**/CT/**/*.png'
        ct_dicom_list = sorted(glob(ct_dcm_paths, recursive=True))
        ct_ground_list = sorted(glob(ct_grd_paths, recursive=True))
    if modality in ('MR', 'ALL'):
        mr_dcm_in = paths[dataset] + '/**/MR/**/InPhase/*.dcm'
        mr_dcm_out = paths[dataset] + '/**/MR/**/OutPhase/*.dcm'
        mr_grd_t1 = paths[dataset] + '/**/MR/**/T1DUAL/**/*.png'
        mr_dcm_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.dcm'
        mr_grd_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.png'
        mr_dcm_in_list = sorted(glob(mr_dcm_in, recursive=True))
        mr_dcm_out_list = sorted(glob(mr_dcm_out, recursive=True))
        mr_dcm_t2_list = sorted(glob(mr_dcm_t2, recursive=True))
        mr_grd_t1_list = sorted(glob(mr_grd_t1, recursive=True))
        mr_grd_t2_list = sorted(glob(mr_grd_t2, recursive=True))
        mr_dicom_list = mr_dcm_in_list + mr_dcm_out_list + mr_dcm_t2_list
        mr_ground_list = mr_grd_t1_list + mr_grd_t1_list + mr_grd_t2_list

    if modality == 'CT':
        if dataset in ('train', 'eval'):
            assert len(ct_dicom_list) == len(ct_ground_list)  # Check lists length
            data_path_list = list(zip(ct_dicom_list, ct_ground_list))
        else:
            data_path_list = list(ct_dicom_list)
    elif modality == 'MR':
        if dataset in ('train', 'eval'):
            assert len(mr_dicom_list) == len(mr_ground_list)  # Check lists length
            data_path_list = list(zip(mr_dicom_list, mr_ground_list))
        else:
            data_path_list = list(mr_dicom_list)
    else:
        if dataset in ('train', 'eval'):
            assert len(ct_dicom_list + mr_dicom_list) == len(ct_ground_list + mr_ground_list)  # Check lists length
            data_path_list = list(zip(ct_dicom_list + mr_dicom_list, ct_ground_list + mr_ground_list))
        else:
            data_path_list = list(ct_dicom_list + mr_dicom_list)
    return data_path_list


if __name__ == '__main__':
    a = get_paths('eval', 'ALL')
    for i in a:
        print(i)