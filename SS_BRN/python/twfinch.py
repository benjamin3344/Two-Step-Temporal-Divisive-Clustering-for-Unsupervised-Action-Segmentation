import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings

from python.read_utils import get_mapping, read_gt_label, estimate_cost_matrix, avg_gt_activity_datasets, get_mapping_new, read_gt_label_new


try:
    from pynndescent import NNDescent#能够正常使用pynndescent时就是True，我是安装了的应该是True

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

# use Approx NN to find first neighbor if samples more than ANN_THRESHOLD
ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine', use_tw_finch=False):#第一次使用的时候构造矩阵，use_tw_finch=True
    s = mat.shape[0]#data,YTI中为261
    
    if initial_rank is not None:#第一次使用orig_dist为None
        orig_dist = []
    elif s <= ANN_THRESHOLD:#样本数小于70000的时候先使用近似网络查找第一邻居，s是帧数，一般都小于70000，就目前的数据集都是小于70000
        if use_tw_finch:
            loc = mat[:, -1]#当传入为data的时候，loc就是time_index，在YTI数据集shape(261,1)
            # mat = mat[:, :-1]    #所有位置的特征        在YTI数据集shape(261,3000)
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2) #在YTI数据集shape(261,261)，时间距离
            #None就是输出这一列所有数据，.T就是转置，矩阵的平方根
            #做了一个按照时序排好的对应各个帧数的矩阵   
            #构造一个矩阵   
            
        else:
            loc_dist = 1.            

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)#计算距离的函数,在YTI数据集shape(261,261),mat是特征距离
        #空间距离用的cos计算，1-cos(x1,x2)
        # orig_dist = orig_dist * loc_dist#在YTI数据集shape(261,261)#这个位置对分数的影响极大
        orig_dist = orig_dist * loc_dist#在YTI数据集shape(261,261)#这个位置对分数的影响极大
        # orig_dist = orig_dist#在YTI数据集shape(261,261)#这个位置对分数的影响极大

        ####
        # orig_dist = orig_dist + 150*loc_dist #在YTI数据集shape(261,261)
        np.fill_diagonal(orig_dist, 1e12)#将对角线元素换为1e12(debug里面显示为0？)
        initial_rank = np.argmin(orig_dist, axis=1)
        un_init, uf_inif = np.unique(initial_rank, return_counts=True)
        #返回二维数组在行方向的最大值，即压缩了列，最后返回是一个（261，1），因为每一行作为一帧考虑的最近邻
    else:#当样本数量大于70000时，第一轮一般都大于70000，后面聚类后会少一些
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
            )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    #csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # 其中data，row_ind和col_ind满足关系a [row_ind [k]，col_ind [k]] = data [k]。
    #也就是只有有数据那一部分是1
    #处理稀疏矩阵(261,261)
    #np.ones_like返回一个用1填充的跟输入 形状和类型 一致的数组
    A = A + sp.eye(s, dtype=np.float32, format='csr')#返回(261,261)的稀疏对角阵。结果的稀疏格式，例如，format=”csr”等。
    A = A @ A.T#@表示的就是矩阵的叉乘，矩阵最普通的乘积方式

    A = A.tolil()
    A.setdiag(0)   #删除掉对角线元素，shape:(12040, 12040)

    return A, orig_dist 

def clust_rank_singlecluster(mat, initial_rank=None, distance='cosine', use_tw_finch=False):#第一次使用的时候构造矩阵，use_tw_finch=True
    s = mat.shape[0]#data,YTI中为261
    
    if initial_rank is not None:#第一次使用orig_dist为None
        orig_dist = []
    elif s <= ANN_THRESHOLD:#样本数小于70000的时候先使用近似网络查找第一邻居，s是帧数，一般都小于70000，就目前的数据集都是小于70000
        if use_tw_finch:
            loc = mat[:, -1]#当传入为data的时候，loc就是time_index，在YTI数据集shape(261,1)
            mat = mat[:, :-1]    #所有位置的特征        在YTI数据集shape(261,3000)
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2) #在YTI数据集shape(261,261)，时间距离
            #None就是输出这一列所有数据，.T就是转置，矩阵的平方根
            #做了一个按照时序排好的对应各个帧数的矩阵   
            #构造一个矩阵   
            
        else:
            loc_dist = 1.            

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)#计算距离的函数,在YTI数据集shape(261,261),mat是特征距离
        
        #空间距离用的cos计算，1-cos(x1,x2)
        orig_dist = orig_dist * loc_dist#在YTI数据集shape(261,261)#这个位置对分数的影响极大
        ####
        # orig_dist = orig_dist + 150*loc_dist #在YTI数据集shape(261,261)
        np.fill_diagonal(orig_dist, 1e12)#将对角线元素换为1e12(debug里面显示为0？)
        initial_rank = np.argmin(orig_dist, axis=1)
        #返回二维数组在行方向的最大值，即压缩了列，最后返回是一个（261，1），因为每一行作为一帧考虑的最近邻
    else:#当样本数量大于70000时，第一轮一般都大于70000，后面聚类后会少一些
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
            )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    #csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # 其中data，row_ind和col_ind满足关系a [row_ind [k]，col_ind [k]] = data [k]。
    #处理稀疏矩阵(261,261)
    #np.ones_like返回一个用1填充的跟输入 形状和类型 一致的数组
    A = A + sp.eye(s, dtype=np.float32, format='csr')#返回(261,261)的稀疏对角阵。结果的稀疏格式，例如，format=”csr”等。
    A = A @ A.T#@表示的就是矩阵的叉乘，矩阵最普通的乘积方式

    A = A.tolil()
    A.setdiag(0)   #删除掉对角线元素，shape:(12040, 12040)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):#a shape:(12040, 12040)
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    #a就是adj是聚类的方法
    #连接组件的数量，返回每个连接组件的标签
    return u, num_clust#u是一个array,num_clust=3772


def cool_mean(M, u):
    s = M.shape[0]#帧数
    un, nf = np.unique(u, return_counts=True)
    #返回去重后的数组以及每个元素出现次数
    #return_index为True时：会构建一个递增的唯一值的新列表，并返回新列表u中的元素在之前定义的旧列表arr中第一次出现值的索引indices
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)#c就是链接组件数量
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()#生成一个非零元素所在的行和列的两个数组,非零元素是1214个，就是比46*46少，里面很多0元素
    r = d[idx]
    v = np.argsort(d[idx])#返回数组从小到大排列的index，这里只考虑了非0元素
    v = v[:2]#只选了最小的两个
    x = [idx[0][v[0]], idx[0][v[1]]]#非零元素的行各选一个
    y = [idx[1][v[0]], idx[1][v[1]]]#非零元素的列格选一个
    s = adj.get_shape()
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1#只选中两个位置作为1，其它全为0
    return a


def req_numclust(c, data, req_clust, distance, use_tw_finch=False):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)#合并，论文里是计算特征平均和时间平均
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_

def req_numclust_singlecluster(c, data, req_clust, distance, use_tw_finch=False):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank_singlecluster(mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data,gt_labels, initial_rank=None, req_clust=None, distance='cosine', tw_finch=True, ensure_early_exit=False, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param tw_finch: Run TW_FINCH on video data.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    if tw_finch:
        n_frames = data.shape[0]
        time_index = (np.arange(n_frames) + 1.) / n_frames #给np.arange整体+1，让time不是从0开始的，而是从1/n_fames开始，是double数据
        #由于arrange是从0开始
        data = np.concatenate([data, time_index[..., np.newaxis]], axis=1) 
        # 将data, time_index[..., np.newaxis]这两个array在axis=1的维度上连接起来，应该是对其进行横向拼接，x变y不变
        #np.newaxis 为 numpy.ndarray（多维数组）增加一个轴，单纯的将[1,1]变为了[[1],[1]]这种增加一个轴
        #拼凑过后，data[,-1]就是time_index
        ensure_early_exit = False
        verbose = False

    # Cast input data to float32
    data = data.astype(np.float32)#数据转化为float32
    
    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance=distance, use_tw_finch=tw_finch)#tw_finch = True，传入的initial_rank是18，50salads是18类
    #adj是什么
    initial_rank = None

    c_new = np.array(range(len(data)),dtype='int32')

    group, num_clust = get_clust(adj, [], min_sim)#连接组件的数量，返回每个连接组件的标签,这步很关键
    c, mat = get_merge([], group, data)#mat shapr(3772,65)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c#261个
    k = 1
    num_clust = [num_clust]#一个数组

    num_clust.insert(0, len(data))

    c = np.column_stack((c_new, c))

    while exit_clust > 1:        
        adj, orig_dist = clust_rank(mat, initial_rank, distance=distance, use_tw_finch=tw_finch)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)#当一个视频类别多的时候继续聚类，初次聚类的类别非常多，几千个类别
        c_, mat = get_merge(c_, u, data)#从3771降到了980再继续降

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))#c shape:(12040, 2),
        exit_clust = num_clust[-2] - num_clust_curr#这就是变化了的exit_clut

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1#感觉k是表示迭代轮数

    #################################################下面是新改的代码
    # ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
    # num_action = num_clust[ind[-1]]
    # # req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_tw_finch=tw_finch)
    # req_c = req_numclust(c[:, ind[-1]], data, num_action, distance, use_tw_finch=tw_finch)

    # # gt_labels, n_labels = read_gt_label(gt_label_path, mapping_dict=mapping_dict)
    # cost_matrix = estimate_cost_matrix(gt_labels, req_c)
    # col_ind_new = np.argmax(-cost_matrix,axis=1)
    # y_pred = col_ind_new[req_c] #这个是结果

    # y_predaction = np.unique(y_pred)
    # num_action = len(y_predaction)

    # cur_acc = metrics.accuracy_score(gt_labels, y_pred)

    # f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro')  # F1-Score
    #     # iou_macro = metrics.jaccard_score(gt_labels, y_pred, average='macro')  # IOU
    #     # penalize equally over/under clustering
    # iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters

    # while(num_action > num_clust):
    #     num_action-=1
    #     ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
    #     req_c = req_numclust(c[:, ind[-1]-1], data, num_action, distance, use_tw_finch=tw_finch)

    #     cost_matrix = estimate_cost_matrix(gt_labels, req_c)
    #     col_ind_new = np.argmax(-cost_matrix,axis=1)
    #     y_pred = col_ind_new[req_c] #这个是结果

    #     y_predaction = np.unique(y_pred)
    #     num_action = len(y_predaction)
    
    # if(num_action < num_clust):
    #     #可以考虑将小的段分出来
    #     req_c
    #####################################################################
    

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            # if(len(ind) == 0):
            #     n = c[:, ind[-1]]
            # else:
            #     n = c[:, ind[-1]]
            n = c[:, ind[-1]]
            # m, indices1, indeices2 = np.unique(n, return_counts=True, return_index=True)
            # req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_tw_finch=tw_finch)
            req_c = req_numclust(n, data, req_clust, distance, use_tw_finch=tw_finch)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',required=True, help='Specify the path to your data csv file.')
    parser.add_argument('--output-path', default="/home/liuyule/FINCH-Clustering-master/results", help='Specify the folder to write back the results.')
    args = parser.parse_args()
    data = np.genfromtxt(args.data_path, delimiter=",").astype(np.float32)
    start = time.time()
    c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True)
    print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

    # Write back
    if args.output_path is not None:
        print('Writing back the results on the provided path ...')
        np.savetxt(args.output_path + '/c.csv', c, delimiter=',', fmt='%d')
        np.savetxt(args.output_path + '/num_clust.csv', np.array(num_clust), delimiter=',', fmt='%d')
        if req_c is not None:
            np.savetxt(args.output_path + '/req_c.csv', req_c, delimiter=',', fmt='%d')
    else:
        print('Results are not written back as the --output-path was not provided')


if __name__ == '__main__':
    main()