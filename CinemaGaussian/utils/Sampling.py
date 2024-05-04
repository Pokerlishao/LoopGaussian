import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.cluster import MiniBatchKMeans


def euclidean_distance_matrix(A, B):
    '''
    Calculate the Euler distance from each point in A to each point in B
    Args:
        A, B: torch.tensor
            A.shape = [num_vertices_A, dim_features]
            B.shape = [num_vertices_B, dim_features]
            The feature dimensions of A and B must be equal
    Returns:
        distances: torch.tensor
            distances.shape = [num_vertices_A, num_vertices_A]
            distances[i][j] means the distance between A[i] and B[j] 
    '''
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)
    AB = torch.matmul(A, B.t())
    distances = A_square - 2 * AB + B_square.t()
    return distances   


def simple_knn(center_point, data, k):
    '''
    Returns:
        k_data.shape = [num_centerPoint, k ,dim_features]
    '''
    distances = euclidean_distance_matrix(center_point[:,:3], data[:,:3])
    _, indices = distances.topk(k, largest=False)
    k_data = data[indices]
    # print(f'k_data {k_data.shape}')
    return k_data
    


def farthest_point_sample(data, npoints):
    """
    Args:
        data: torch.rensor shape = [num_veritices, dim]
        npoints: the number points want to sample
    Returns:
        data.shape = [npoints, dim]
    """
    N,D = data.shape 
    xyz = data[:,:3] #position
    centroids = torch.zeros(size=(npoints,), device=data.device)
    dictance = torch.ones(size=(N,),device=data.device)*1e10
    farthest = torch.randint(low=0,high=N,size=(1,),device=data.device)
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        farthest = torch.argmax(dictance,dim=-1)
    data= data[centroids.type(torch.long)]
    return data



def kmeans(tensor, num_clusters = 50, num_iterations=500):
    device = tensor.device
    data_np = tensor.cpu().numpy()
    batch_size = num_clusters * 2
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, max_iter=num_iterations, random_state=0, n_init=10)
    kmeans.fit(data_np)
    # centroids = torch.tensor(kmeans.cluster_centers_)
    labels = kmeans.predict(data_np)
    clustered_data = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        clustered_data[label].append(tensor[i])
        # clustered_data[label].append(i)  ## index
    clustered_data = [torch.stack(cluster).to(device) for cluster in clustered_data]
    return clustered_data


def densification_interval(sparse_points, desired_resolution=5):
    sparse_points = sparse_points.unsqueeze(0).permute(0,2,1) 
    dense_points = F.interpolate(sparse_points, scale_factor=desired_resolution, mode='linear', align_corners=True).squeeze().permute(1,0)
    return dense_points