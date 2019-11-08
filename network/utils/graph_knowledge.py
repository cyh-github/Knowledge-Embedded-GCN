import numpy as np

class k_Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - dastance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'sbu_edge':
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(6, 5), (5, 4), (9, 8), (8, 7), (12, 11), (11, 10), (15, 14),
                              (14, 13), (10, 3), (13, 3), (4, 3), (7, 3), (2, 3), (1, 2)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 3

        elif layout == 'sbu_edge_2r':
            self.num_node = 30
            self_link = [(i, i) for i in range(self.num_node)]
            
            #relation org
            relation_base = [(6, 21), (9, 24), (6, 24), (9, 21),
                             (3, 18), 
                             (6, 16), (21, 1),
                             (27, 3), (12, 18), (30, 3), (15, 18),
                             (21, 3), (24, 3), (6, 18), (9, 18), 
                             (21, 7), (24, 4),  (6, 22), (9, 19),
                             (9, 22), (21, 4), (24, 4), (6, 19)
                             ]
                             
            self.relation_link = [(i - 1, j - 1) for (i, j) in relation_base]
            self.edge = self_link + self.relation_link
            self.center = 3-1

        elif layout == 'sbu_part_2r':
            self.num_node = 10  # two person 10 parts
            self_link = [(i, i) for i in range(self.num_node)]


            relation_link = [(1, 10), (3, 10), (2, 10), (4, 10),  # person 1 to another person tunck
                             (6, 5), (7, 5), (8, 5), (9, 5),  # person 2 to another person tunck
                             (5, 10),  # two tunck
                             (1, 6), (3, 6), (1, 8), (3, 8)]  # two arms
            self.relation_link = [(i - 1, j - 1) for (i, j) in relation_link]

            self.edge = self_link  + self.relation_link
            self.center = 4  # person tunck

        elif layout == 'ntu_edge_2r':
            self.num_node = 50
            self_link = [(i, i) for i in range(self.num_node)]
            
            Krelation_base = [(24, 29), (22, 49), (24, 47), (22, 47), (2, 27), 
                             (24, 49), (49, 4), (24, 27), (22, 27), (49, 2),
                             (47, 2), (24, 34), (24, 30), (22, 34), (22, 30),
                             (49, 9), (49, 5), (47, 9), (47, 5), (20, 27), 
                             (16, 27), (45, 2), (41, 2)]
            

            self.relation_link = [(i - 1, j - 1) for (i, j) in Krelation_base]
            
            self.edge = self_link + self.relation_link
            self.center = 2-1
        elif layout == 'ntu_part_2r':
            self.num_node = 16 #two person 16 parts
            self_link = [(i, i) for i in range(self.num_node)]
            relation_link = [(5,10),(5,13),(5,9),(5,12),(5,15),(5,16), #person 1 right hand to another person body
                             (2,10),(2,13),(2,9),(2,12),(2,15),(2,16),#person 1 left hand to another person body
                             (8,16),#two tunck
                             (3,16),(6,16),(11,8),(14,8),#foot to tunck
                             (10,4),(10,1),(10,7),(10,8),#person 2 left hand to another person body
                             (13,4),(13,1),(13,7),(13,8)]#person 2 right hand to another person body
            self.relation_link = [(i - 1, j - 1) for (i, j) in relation_link]


            self.edge = self_link + self.relation_link
            self.center = 7
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
