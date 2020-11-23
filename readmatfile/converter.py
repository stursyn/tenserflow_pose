import scipy.io as sio
from tqdm import tqdm

mat = sio.loadmat("mpii_human_pose_v1_u12_1.mat", struct_as_record=False)

print(mat['RELEASE'].shape)
matlab_mpii = mat['RELEASE'][0, 0]
num_images = annotation_mpii = matlab_mpii.__dict__['annolist'][0].shape[0]

#num_images = 1

initial_index = 0
batch = 200
while initial_index < num_images:
    # Initialize empty placeholder
    img_dict = {'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': []}}

    # Iterate over each image
    for img_idx in tqdm(range(initial_index, min(initial_index + batch, num_images))):
        annotation_mpii = matlab_mpii.__dict__['annolist'][0, img_idx]
        train_test_mpii = matlab_mpii.__dict__['img_train'][0, img_idx].flatten()[0]
        person_id = matlab_mpii.__dict__['single_person'][img_idx][0].flatten()

        # Load the individual image. Throw an exception if image corresponding to filename not available.
        img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]
        if img_name == '015601864.jpg':
            print(img_name)
            print(annotation_mpii)
            print(train_test_mpii)
            print(person_id)
            for person in range(len(person_id)):
                print('X1:',annotation_mpii.__dict__['annorect'][0, person].__dict__['x1'][0])
                print('Y1:',annotation_mpii.__dict__['annorect'][0, person].__dict__['y1'][0])
                print('X2:',annotation_mpii.__dict__['annorect'][0, person].__dict__['x2'][0])
                print('Y2:',annotation_mpii.__dict__['annorect'][0, person].__dict__['y2'][0])
                print('scale:',annotation_mpii.__dict__['annorect'][0, person].__dict__['scale'][0])
                calcCenter = 0

                for point in annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0].__dict__['point'][0]:
                    print('x',point.__dict__['x'][0])

                    calcCenter += point.__dict__['x'][0, 0]
                    print('y',point.__dict__['y'][0])
                    print('id',point.__dict__['id'][0])
                    print('is_vis',point.__dict__['is_visible'])

                print(calcCenter/len(annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0].__dict__['point'][0]))
                print(len(annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0].__dict__['point'][0]))

    initial_index += batch
