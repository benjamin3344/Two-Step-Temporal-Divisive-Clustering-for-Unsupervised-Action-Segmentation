import numpy as np
import pandas as pd


def get_mapping(maping_file_path):
    # fr = open(maping_file_path)
    # mapping_dict = {}
    # keys = []
    # for line in fr:
    #     v = line.replace("\n","").split(" ")
    #     s = ""
    #     for i in range(len(v)):
    #         if(i >= 1):
    #             s += v[i] + " "
    #     s = s[:-1]
    #     mapping_dict[v[0]] = s
    # fr.close()
    # mapping_dict = {value:key for key,value in mapping_dict.items()}
    df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
    mapping_dict = dict(zip(df['action_name'], df['index']))
    return mapping_dict


def read_gt_label(gt_label_path, mapping_dict=None):
    df_gt = pd.read_csv(gt_label_path, sep=" ", header=None)
    gt = df_gt[0].tolist()
    if mapping_dict is not None:
        gt_label = [mapping_dict[i] for i in gt]#读取的每一帧对应的gt,转为数字
        # np.savetxt("/home/liuyule/FINCH-Clustering-master/labels.txt",gt_label)
        gt_label = np.array(gt_label)#转化为数字对应的npy文件
        n_labels = len(mapping_dict)#返回labels的数量，就是自动做的数量
    else:
        _, gt_label = np.unique(gt, return_inverse=True)
        n_labels = len(np.unique(gt_label))

    # make sure gt label do not contain -ve entries
    gt_min = np.min(gt_label)
    if gt_min < 0:
        gt_label = gt_label - gt_min

    return gt_label, n_labels


def get_mapping_new(maping_file_path):
    df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
    mapping_dict = dict(zip(df['index'], df['action_name']))
    return mapping_dict


def read_gt_label_new(gt_label_path, mapping_dict=None):
    df_gt = pd.read_csv(gt_label_path, sep=" ", header=None)
    gt = df_gt[0].tolist()
    if mapping_dict is not None:
        gt_label = [mapping_dict[i] for i in gt]#读取的每一帧对应的gt,转为数字
        gt_label = np.array(gt_label)#转化为数字对应的npy文件
        n_labels = len(mapping_dict)#返回labels的数量，就是自动做的数量
    else:
        _, gt_label = np.unique(gt, return_inverse=True)
        n_labels = len(np.unique(gt_label))

    # make sure gt label do not contain -ve entries
    gt_min = np.min(gt_label)
    if gt_min < 0:
        gt_label = gt_label - gt_min

    return gt_label, n_labels


def estimate_cost_matrix(gt_labels, cluster_labels):#cluster_labels就是req_c，由FINCH得到的
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_labels)#去掉重复元素，并按从小到大排序
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)#由于L_gt从0开始的
    profit_mat = np.zeros((nClass_pred, dim_1))
    for i in L_pred:
        idx = np.where(cluster_labels == i)#将符合条件的元素坐标进行返回，一个类一个类的判断，每个在pred类中的动作所在预测中的位置
        gt_selected = gt_labels[idx]#真实的一个预测段的内部所有动作，预测动作所在的位置的真实动作
        n, indices = np.unique(gt_selected, return_index=True)
        for j in L_gt:
            m = (gt_selected == j)#一个true，false数组
            profit_mat[i][j] = np.count_nonzero(gt_selected == j)#统计非0个数，即统计True的个数
            #在预测动作i的片段中gt的动作片段有多少个
    return -profit_mat


# Pre_computed average # of actions per activity in the datasets. We cluster each video to this K
avg_gt_activity_datasets = {
                            # 'Breakfast': {'salat': 5},
                            'Breakfast': {'cereals': 4, 'coffee': 4, 'friedegg': 6, 'juice': 5,
                                          'milk': 4, 'pancake': 9, 'salat': 5,
                                          'sandwich': 5, 'scrambledegg': 7, 'tea': 4},
                            '50salads': {'features': 18},
                            'Hollywood_extended': {'features': 3},
                            'YTI': {'changing_tire': 10, 'coffee': 8, 'cpr': 6, 'jump_car': 10, 'repot': 7},
                            'MPII_Cooking': {'d01': 14.0, 'd02': 24.0, 'd03': 34.0, 'd04': 36.0,
                                             'd05': 33.0, 'd06': 20.0, 'd07': 34.0, 'd08': 33.0,
                                             'd09': 21.0, 'd10': 22.0, 'd11': 22.0, 'd12': 17.0,
                                             'd13': 22.0, 'd14': 18.0, 'd21': 14.0, 'd23': 15.0,
                                             'd24': 12.0, 'd25': 9.0, 'd26': 23.0, 'd27': 20.0,
                                             'd28': 14.0, 'd29': 14.0, 'd31': 14.0, 'd32': 19.0,
                                             'd34': 12.0, 'd35': 12.0, 'd36': 14.0, 'd39': 21.0,
                                             'd40': 12.0, 'd41': 18.0, 'd42': 25.0, 'd43': 17.0,
                                             'd45': 13.0, 'd46': 16.0, 'd47': 24.0, 'd48': 10.0,
                                             'd49': 22.0, 'd50': 9.0, 'd51': 17.0, 'd52': 12.0,
                                             'd53': 15.0, 'd54': 15.0, 'd55': 11.0, 'd57': 36.0,
                                             'd58': 15.0, 'd59': 27.0, 'd60': 14.0, 'd61': 4.0,
                                             'd62': 10.0, 'd63': 16.0, 'd65': 11.0, 'd67': 7.0,
                                             'd68': 12.0, 'd69': 24.0, 'd70': 15.0, 'd71': 25.0,
                                             'd72': 7.0, 'd73': 23.0, 'd74': 21.0}}
