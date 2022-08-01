#%%
import torch
import networkx as nx

class SuperVertex(object):
	def __init__(self, node_feat, edge_index, edge_type=None):
		self.node_feat = node_feat
		self.edge_index = edge_index
		self.edge_type = edge_type

		# get the number of nodes and edges
		self.n_node = node_feat.shape[0]
		self.n_edge = edge_index.shape[1]

		# initialize in vertex and out vertex lists
		self.in_vertex_list = []
		self.out_vertex_list = []
		self.if_start_vertex = True

		self.__process_edges__()

	def __process_edges__(self):
		if edge_type is None:
			self.if_multi_relational = False
			self.n_edge_type = None
		else:
			self.if_multi_relational = True	
			self.n_edge_type = edge_type.unique().shape[0]

	def __repr__(self) -> str:
		return f'SuperVertex(\n node_feat={self.node_feat.shape}, \n edge_index={self.edge_index.shape}, \n edge_type={self.edge_type.shape}, \n if_multi_relational={self.if_multi_relational}, \n n_edge_type={self.n_edge_type})'

	def add_in_vertex(self, vertex_name: str):
		self.in_vertex_list.append(vertex_name)

	def add_out_vertex(self, vertex_name: str):
		self.out_vertex_list.append(vertex_name)

	def set_name(self, vertex_name: str):
		self.name = vertex_name


class SuperGraph(object):
	def __init__(self, svertex_dict: dict, sedge_dict: dict):
		self.svertex_dict = svertex_dict
		self.sedge_dict = sedge_dict

		self.__process_graph__()
		self.__process_svertex__()

	def __repr__(self) -> str:
		return f'SuperGraph(\n svertex_dict={self.svertex_dict.keys()}, \n sedge_dict={self.sedge_dict.keys()}, \n G={self.G}), \n topological_order={self.topological_order}'

	def __process_graph__(self):
		self.G = nx.DiGraph()
		self.G.add_edges_from(self.sedge_dict.keys())

		assert nx.is_directed_acyclic_graph(self.G), 'The supergraph is not a directed acyclic graph.'

		self.n_svertex = self.G.number_of_nodes()
		self.n_sedge = self.G.number_of_edges()
		self.topological_order = list(nx.topological_sort(self.G))

	def __process_svertex__(self):
		for n1, n2 in self.G.edges():
			self.svertex_dict[n2].add_in_vertex(n1)
			self.svertex_dict[n1].add_out_vertex(n2)

		for n, sv in self.svertex_dict.items():
			sv.set_name(n)
			if sv.in_vertex_list:
				sv.if_start_vertex = False

node_feat = torch.randn(4, 10)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
edge_type = torch.tensor([0, 0, 1, 1])

sv1 = SuperVertex(node_feat, edge_index, edge_type)
sv2 = SuperVertex(node_feat, edge_index, edge_type)
sv3 = SuperVertex(node_feat, edge_index, edge_type)

sg = SuperGraph(
	svertex_dict={'sv1': sv1, 'sv2': sv2, 'sv3': sv3},
	sedge_dict={
		('sv1', 'sv3'): torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]]), 
		('sv2', 'sv3'): torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])}
)





# %%
