import numpy as np
import math
import queue


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):  # 使用线性学习率
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def bfs(graph, node, book, class_id):
    book[node] = class_id
    q = queue.Queue()
    q.put(node)
    while not q.empty():
        cur_node = q.get()
        for next_node in graph[cur_node]:
            if next_node == cur_node or book[next_node] != -1:
                continue
            book[next_node] = class_id
            q.put(next_node)


def getClassFromGraphs(graphs: np.array):
    for row_index, row in enumerate(graphs):
        graph = []
        for i in range(len(row)):
            graph.append([])
        for col_index in range(len(row)):
            graph[col_index].append(row[col_index])
            graph[row[col_index]].append(col_index)

        book = []
        for i in range(len(row)):
            book.append(-1)
        class_id = 0
        for node in range(len(row)):
            if book[node] == -1:
                bfs(graph, node, book, class_id)
                class_id += 1
        for col_index in range(len(row)):
            graphs[row_index][col_index] = book[col_index]
