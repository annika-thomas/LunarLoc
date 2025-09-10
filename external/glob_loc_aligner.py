import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import matplotlib.pyplot as plt
import clipperpy
import matplotlib.pyplot as plt
from utils import load_mission_data, generateAssociationList, rigid_transform_3D, generate_alignment_points, extract_rock_locations
import numpy as np
from scipy.spatial.transform import Rotation as R

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



gt_rock_locations = np.array(extract_rock_locations("/home/annika/Documents/LAC_LunarLoc_Data/Preset_1.xml"))

# only keep rocks within x from -14 to 2 and y from -12 to 0
gt_rock_locations = gt_rock_locations[(gt_rock_locations[:, 0] >= -5) & (gt_rock_locations[:, 0] <= 20) &
                                      (gt_rock_locations[:, 1] >= -11) & (gt_rock_locations[:, 1] <= -1)]

print("ground truth rock locations shape:", gt_rock_locations.shape)
print("ground truth rock locations:", gt_rock_locations)

np.random.seed(3)


mission_data1 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial1.csv')
mission_data2 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial2.csv')
mission_data3 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial3.csv')
mission_data4 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial4.csv')
mission_data5 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial5.csv')
mission_data6 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial6.csv')
mission_data7 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial7.csv')
mission_data8 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial8.csv')
mission_data9 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial9.csv')
mission_data10 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial10.csv')
mission_data11 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial11.csv')
mission_data12 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial12.csv')
mission_data13 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial13.csv')
mission_data14 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial14.csv')
mission_data15 = load_mission_data('/home/annika/Documents/LAC_LunarLoc_Data/trial15.csv')


poses1 = np.array([pose[:3, 3] for pose, _ in mission_data1])
poses2 = np.array([pose[:3, 3] for pose, _ in mission_data2])
poses3 = np.array([pose[:3, 3] for pose, _ in mission_data3])
poses4 = np.array([pose[:3, 3] for pose, _ in mission_data4])
poses5 = np.array([pose[:3, 3] for pose, _ in mission_data5])
poses6 = np.array([pose[:3, 3] for pose, _ in mission_data6])
poses7 = np.array([pose[:3, 3] for pose, _ in mission_data7])
poses8 = np.array([pose[:3, 3] for pose, _ in mission_data8])
poses9 = np.array([pose[:3, 3] for pose, _ in mission_data9])
poses10 = np.array([pose[:3, 3] for pose, _ in mission_data10])
poses11 = np.array([pose[:3, 3] for pose, _ in mission_data11])
poses12 = np.array([pose[:3, 3] for pose, _ in mission_data12])
poses13 = np.array([pose[:3, 3] for pose, _ in mission_data13])
poses14 = np.array([pose[:3, 3] for pose, _ in mission_data14])
poses15 = np.array([pose[:3, 3] for pose, _ in mission_data15])




# # Plot poses and boulders
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# # Plot poses
# ax.scatter(poses1[:, 0], poses1[:, 1], poses1[:, 2], c='r', marker='o', label='Path 1')
# ax.scatter(poses2[:, 0], poses2[:, 1], poses2[:, 2], c='b', marker='o', label='Path 2')
ax.plot(poses1[:, 0], poses1[:, 1], poses1[:, 2], 'r-', alpha=0.5, label='Path 1')
ax.plot(poses2[:, 0], poses2[:, 1], poses2[:, 2], 'b-', alpha=0.5, label='Path 2')
ax.plot(poses3[:, 0], poses3[:, 1], poses3[:, 2], 'g-', alpha=0.5, label='Path 3')
ax.plot(poses4[:, 0], poses4[:, 1], poses4[:, 2], 'y-', alpha=0.5, label='Path 4')
ax.plot(poses5[:, 0], poses5[:, 1], poses5[:, 2], 'c-', alpha=0.5, label='Path 5')
ax.plot(poses6[:, 0], poses6[:, 1], poses6[:, 2], 'm-', alpha=0.5, label='Path 6')
ax.plot(poses7[:, 0], poses7[:, 1], poses7[:, 2], 'k-', alpha=0.5, label='Path 7')
ax.plot(poses8[:, 0], poses8[:, 1], poses8[:, 2], 'orange', alpha=0.5, label='Path 8')
ax.plot(poses9[:, 0], poses9[:, 1], poses9[:, 2], 'purple', alpha=0.5, label='Path 9')
ax.plot(poses10[:, 0], poses10[:, 1], poses10[:, 2], 'brown', alpha=0.5, label='Path 10')
ax.plot(poses11[:, 0], poses11[:, 1], poses11[:, 2], 'pink', alpha=0.5, label='Path 11')
ax.plot(poses12[:, 0], poses12[:, 1], poses12[:, 2], 'gray', alpha=0.5, label='Path 12')
ax.plot(poses13[:, 0], poses13[:, 1], poses13[:, 2], 'olive', alpha=0.5, label='Path 13')
ax.plot(poses14[:, 0], poses14[:, 1], poses14[:, 2], 'teal', alpha=0.5, label='Path 14')
ax.plot(poses15[:, 0], poses15[:, 1], poses15[:, 2], 'navy', alpha=0.5, label='Path 15')


boulders1 = mission_data1[-1][1]  # Get boulders from last frame
boulders2 = mission_data2[-1][1]  # Get boulders from last frame
boulders3 = mission_data3[-1][1]  # Get boulders from last frame
boulders4 = mission_data4[-1][1]  # Get boulders from last frame
boulders5 = mission_data5[-1][1]  # Get boulders from last frame
boulders6 = mission_data6[-1][1]  # Get boulders from last frame
boulders7 = mission_data7[-1][1]  # Get boulders from last frame
boulders8 = mission_data8[-1][1]  # Get boulders from last frame
boulders9 = mission_data9[-1][1]  # Get boulders from last frame
boulders10 = mission_data10[-1][1]  # Get boulders from last frame
boulders11 = mission_data11[-1][1]  # Get boulders from last frame
boulders12 = mission_data12[-1][1]  # Get boulders from last frame
boulders13 = mission_data13[-1][1]  # Get boulders from last frame
boulders14 = mission_data14[-1][1]  # Get boulders from last frame
boulders15 = mission_data15[-1][1]  # Get boulders from last frame


if len(boulders1) > 0:
    ax.scatter(boulders1[:, 0], boulders1[:, 1], boulders1[:, 2], 
               c='orange', marker='^', s=100, label='Boulders 1')
if len(boulders2) > 0:
    ax.scatter(boulders2[:, 0], boulders2[:, 1], boulders2[:, 2], 
               c='green', marker='^', s=100, label='Boulders 2')
if len(boulders3) > 0:
    ax.scatter(boulders3[:, 0], boulders3[:, 1], boulders3[:, 2], 
               c='red', marker='^', s=100, label='Boulders 3')
if len(boulders4) > 0:
    ax.scatter(boulders4[:, 0], boulders4[:, 1], boulders4[:, 2], 
               c='green', marker='^', s=100, label='Boulders 4')
if len(boulders5) > 0:
    ax.scatter(boulders5[:, 0], boulders5[:, 1], boulders5[:, 2], 
               c='blue', marker='^', s=100, label='Boulders 5')
if len(boulders6) > 0:
    ax.scatter(boulders6[:, 0], boulders6[:, 1], boulders6[:, 2], 
               c='brown', marker='^', s=100, label='Boulders 6')
if len(boulders7) > 0:
    ax.scatter(boulders7[:, 0], boulders7[:, 1], boulders7[:, 2], 
               c='yellow', marker='^', s=100, label='Boulders 7')
if len(boulders8) > 0:
    ax.scatter(boulders8[:, 0], boulders8[:, 1], boulders8[:, 2], 
               c='olive', marker='^', s=100, label='Boulders 8')
if len(boulders9) > 0:
    ax.scatter(boulders9[:, 0], boulders9[:, 1], boulders9[:, 2], 
               c='teal', marker='^', s=100, label='Boulders 9')
if len(boulders10) > 0:
    ax.scatter(boulders10[:, 0], boulders10[:, 1], boulders10[:, 2], 
               c='navy', marker='^', s=100, label='Boulders 10')
if len(boulders11) > 0:
    ax.scatter(boulders11[:, 0], boulders11[:, 1], boulders11[:, 2], 
               c='orange', marker='^', s=100, label='Boulders 11')
if len(boulders12) > 0:
    ax.scatter(boulders12[:, 0], boulders12[:, 1], boulders12[:, 2], 
               c='purple', marker='^', s=100, label='Boulders 12')
if len(boulders13) > 0:
    ax.scatter(boulders13[:, 0], boulders13[:, 1], boulders13[:, 2], 
               c='blue', marker='^', s=100, label='Boulders 13')
if len(boulders14) > 0:
    ax.scatter(boulders14[:, 0], boulders14[:, 1], boulders14[:, 2], 
               c='purple', marker='^', s=100, label='Boulders 14')
if len(boulders15) > 0:
    ax.scatter(boulders15[:, 0], boulders15[:, 1], boulders15[:, 2], 
               c='pink', marker='^', s=100, label='Boulders 15')

# plot the ground truth rock locations
ax.scatter(gt_rock_locations[:, 0], gt_rock_locations[:, 1], gt_rock_locations[:, 2],
           c='black', marker='*', s=200, label='Ground Truth Rocks')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Pose and boulder locations in world frames')
plt.show()

# exit(0)

import csv
headers = ['trav1', 'trav2', 'num_boulders1', 'num_boulders2', 'num_associations', 'translation error', 'rot error (deg)', 'rot error (rad)']
rows = []

print("adding mission data to list")
mission_data = []
for i in range(1, 16):
    mission_data.append(eval(f'mission_data{i}'))

for countouter in range(1, 16):
    for countinner in range(1, 16):
        if countouter == countinner:
            continue

        print(f"countouter: {countouter-1}, countinner: {countinner-1}")

        # traverse_1_data = gt_rock_locations
        traverse_2_data = mission_data[countinner-1]

        boulder_locations1 = gt_rock_locations
        boulder_locations2 = traverse_2_data[-1][1] 

        if len(boulder_locations1) * len(boulder_locations2) > 22000:  # or some reasonable threshold
            print("Too big, skipping")
            continue

        print("boulder locations 1 shape:", boulder_locations1.shape)
        print("boulder locations 2 shape:", boulder_locations2.shape)

        A = generateAssociationList(len(boulder_locations1), len(boulder_locations2))

        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.epsilon = 0.5
        iparams.sigma = 0.5 * iparams.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)

        params = clipperpy.Params()
        params.rounding = clipperpy.Rounding.DSD_HEU
        clipper = clipperpy.CLIPPER(invariant, params)

        numAssocRows, _ = A.shape

        inliers = np.zeros(numAssocRows,dtype=bool)
        weights = np.zeros(numAssocRows,dtype=float)

        clipper.score_pairwise_consistency(boulder_locations1.T, boulder_locations2.T, A)

        clipper.solve()
        Ain = clipper.get_selected_associations()

        print("Ain shape:", Ain.shape)


        # # plot the center points of the masks
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(boulder_locations1[:, 0], boulder_locations1[:, 1], boulder_locations1[:, 2], c='r', marker='o')
        # ax.scatter(boulder_locations2[:, 0], boulder_locations2[:, 1], boulder_locations2[:, 2], c='b', marker='o')
        # plt.axis('equal')
        # plt.title('Mask centers in world frames')

        associatedPointLocations_1 = boulder_locations1[Ain[:,0]]
        associatedPointLocations_2 = boulder_locations2[Ain[:,1]]

        # associatedPointLocations_w_1 = boulder_locations2[Ain[:,0]]    
        # associatedPointLocations_w_2 = boulder_locations2[Ain[:,1]]

        # plot the associated points with lines connecting them
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(associatedPointLocations_1[:, 0], associatedPointLocations_1[:, 1], associatedPointLocations_1[:, 2], c='r', marker='o')
        # ax.scatter(associatedPointLocations_2[:, 0], associatedPointLocations_2[:, 1], associatedPointLocations_2[:, 2], c='b', marker='o')

        # for i in range(len(associatedPointLocations_1)):
        #     ax.plot([associatedPointLocations_1[i, 0], associatedPointLocations_2[i, 0]],
        #         [associatedPointLocations_1[i, 1], associatedPointLocations_2[i, 1]],
        #         [associatedPointLocations_1[i, 2], associatedPointLocations_2[i, 2]], 'k-')

        # plt.axis('equal')
        # plt.title('Associated mask centers in world frames')
        # plt.show()


        R_est, t_est = rigid_transform_3D(associatedPointLocations_2.T, associatedPointLocations_1.T)

        # flatten t estimate
        t_est = t_est.flatten()

        T_est = np.eye(4)
        T_est[:3, :3] = R_est
        T_est[:3, 3] = t_est

        print("R estimated:", R_est)
        print("t estimated:", t_est)

        # print error from identity in rotation and translation
        R_error = np.dot(R_est.T, R_est)
        # Compute the trace of the relative rotation matrix
        trace_R_error = np.trace(R_error)
        # Compute the rotation error in radians, ensuring the input to arccos is within [-1, 1]
        theta_error = np.arccos(np.clip((trace_R_error - 1) / 2, -1.0, 1.0))
        # Convert to degrees if needed
        theta_error_deg = np.degrees(theta_error)
        print(f"Rotation error (in radians): {theta_error}")
        print(f"Rotation error (in degrees): {theta_error_deg}")
        # Compute the translation error
        translation_error = np.linalg.norm(t_est)
        print(f"Translation error {countouter} and {countinner}: {translation_error}")

        # headers = ['trav1', 'trav2', 'num_boulders1', 'num_boulders2', 'num_associations', 'translation error', 'rot error (deg)', 'rot error (rad)']
        row = [0, countinner, len(boulder_locations1), len(boulder_locations2), Ain.shape[0], translation_error, theta_error_deg, theta_error]
        rows.append(row)

with open('clipper_output_glob_loc.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers first
    writer.writerows(rows)    # Then all rows

# true_traj_path = '/home/annika/Gaussian-SLAM/scripts/data/Replica-SLAM/room0/traj.txt'
# true_traj_path1 = '/media/annika/Extreme SSD/Kimera-Multi/data/outdoor/sparkal1/sparkal1_gt_odom.csv'
# true_traj_path2 = '/media/annika/Extreme SSD/Kimera-Multi/data/outdoor/sparkal2/sparkal2_gt_odom.csv'
# true_traj1 = read_odom_poses_from_csv(true_traj_path1)
# true_traj2 = read_odom_poses_from_csv(true_traj_path2)
# true_traj1 = dataset1.poses
# true_traj2 = dataset2.poses
# print("true traj 1 shape:", len(true_traj1))
# print("true traj 2 shape:", len(true_traj2))
# # true_traj = poses_gt

# # print("true traj 1 shape:", len(true_traj1))
# # print("true traj 2 shape:", len(true_traj2))

# print("keyframe id 1: ", keyframe_ids1[0])
# print("keyframe id 2: ", keyframe_ids2[0])

# # traj at keypoint1
# true_pose_keyframe1 = true_traj1[keyframe_ids1[0]]
# true_pose_keyframe2 = true_traj2[keyframe_ids2[0]]

# # print("true_pose_keyframe1 shape:", true_pose_keyframe1)
# # print("true_pose_keyframe2 shape:", true_pose_keyframe2)

# print(keyframe_ids1[0])
# print(keyframe_ids2[0])
# print("true pose 1:", true_pose_keyframe1)
# print("estimated pose 1:", keyframe_c2w_1)
# print("true pose 2:", true_pose_keyframe2)
# print("estimated pose 2:", keyframe_c2w_2)

# keyframe_c2w_1 = np.reshape(keyframe_c2w_1, (4, 4))
# keyframe_c2w_2 = np.reshape(keyframe_c2w_2, (4, 4))

# T_kf_1_inv = np.linalg.inv(keyframe_c2w_1)
# T_kf_2 = keyframe_c2w_2
# T_gslam_12 = np.dot(T_kf_1_inv, T_kf_2)

# # reshape to 4x4
# true_pose_keyframe1 = np.reshape(true_pose_keyframe1, (4, 4))
# true_pose_keyframe2 = np.reshape(true_pose_keyframe2, (4, 4))

# # print(true_pose_keyframe1)

# # transform from true traj 1 to true traj 2 poses
# T_A_inv = np.linalg.inv(true_pose_keyframe1)
# T_B = true_pose_keyframe2
# T_AB = np.dot(T_A_inv, T_B)

# # Extract rotation (top-left 3x3 part of the matrix)
# rotation_AB = T_AB[:3, :3]

# # Extract translation (top-right 3x1 part of the matrix)
# translation_AB = T_AB[:3, 3]

#     # Compute the trace of the relative rotation matrix
# trace_R = np.trace(rotation_AB)

# # Compute the rotation error in radians
# theta_rad = np.arccos((trace_R - 1) / 2)

# # Convert to degrees if needed
# theta_deg = np.degrees(theta_rad)

# translation = np.linalg.norm(translation_AB)

# # print("true rotation:", rotation_AB)
# # print("true translation:", translation_AB)

# # Compute the relative rotation matrix: R_error = R_true^T * R_est
# R_error = np.dot(rotation_AB.T, R_est)

# # Compute the trace of the relative rotation matrix
# trace_R_error = np.trace(R_error)

# # Compute the rotation error in radians
# theta_error = np.arccos((trace_R_error - 1) / 2)

# # Convert to degrees if needed
# theta_error_deg = np.degrees(theta_error)

# # Output the result
# # print(f"Rotation error (in radians): {theta_error}")
# # print(f"Rotation error (in degrees): {theta_error_deg}")

# print("translation_ab = ", translation_AB)
# print("t_est = ", t_est)
# print("translation error = ", np.linalg.norm(translation_AB + t_est))

# # Compute the translation error
# translation_error = np.linalg.norm(translation_AB + t_est)
# # print(f"Translation error: {translation_error}")

# output_data.append([keyframe_ids1[0], keyframe_ids2[0], len(masks1), len(masks2), Ain.shape[0], T_AB, T_gslam_12, T_est, translation, theta_deg, translation_error, theta_error_deg, theta_error])
# # Print total memory allocated by tensors in bytes
# print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

# # Print total reserved (or cached) memory by the allocator in bytes
# print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

# # Create a pandas DataFrame from the list of rows
# df = pd.DataFrame(output_data, columns=columns)

# # save to output folder
# # output_path = "/home/annika/Gaussian-SLAM/output/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household"
# output_path = f"/media/annika/Extreme SSD/output/Kimera/{test}/"
# df.to_csv(output_path + f'submap_clipper_output_all_{countouter}.csv', index=False)

# output_data_all.append(output_data)

# print(f"CSV file 'submap_clipper_output_all_{countouter}.csv' has been created successfully.")

# df_all = pd.DataFrame(output_data_all, columns=columns)
# df_all.to_csv(output_path + 'submap_clipper_output_all.csv', index=False)