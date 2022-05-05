"""
graph.py
----------------
Graph Supporting Data Structure and Functions:
@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   06/01/2020
"""


class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class UnionFind:
    def __init__(self, nodes):
        self.father = {}
        for i in range(len(nodes)):
            self.father[nodes[i].label] = nodes[i].label

    def find(self, x):
        if self.father[x] == x:
            return x
        else:
            self.father[x] = self.find(self.father[x])
            return self.father[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y


class UndirectedGraph:
    """
    @param: nodes: a array of Undirected graph node
    @return: a connected set of a Undirected graph
    """

    def connectedSet(self, nodes):
        uf = UnionFind(nodes)
        for node in nodes:
            for neighbor in node.neighbors:
                uf.union(node.label, neighbor.label)

        hash = {}
        for node in nodes:
            root_label = uf.find(node.label)
            if root_label not in hash:
                hash[root_label] = []
            hash[root_label].append(node.label)

        res = []
        for _, node in hash.items():
            res.append(node)

        return list(map(sorted, list(map(set, res))))
