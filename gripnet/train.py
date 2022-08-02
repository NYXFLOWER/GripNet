#%%
import torch
import networkx as nx
from layers import GripNetRGCN, GripNetGCN

# set random seeds
torch.manual_seed(1111)
# np.random.seed(1111)

class superVertex(object):
	def __init__(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor=None, edge_weight: torch.Tensor=None):
		self.node_feat = node_feat
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.edge_weight = edge_weight

		# get the number of nodes, node features and edges
		self.n_node, self.n_node_feat = node_feat.shape
		self.n_edge = edge_index.shape[1]

		# initialize in vertex and out vertex lists
		self.in_svertex_list = []
		self.out_svertex_list = []
		self.if_start_svertex = True

		self.__process_edges__()

	def __process_edges__(self):
		if edge_type is None:
			# self.if_multi_relational = False
			self.n_edge_type = 1
		else:
			# self.if_multi_relational = True	
			self.n_edge_type = edge_type.unique().shape[0]

	def __repr__(self) -> str:
		return f'SuperVertex(\n node_feat={self.node_feat.shape}, \n edge_index={self.edge_index.shape}, \n edge_type={self.edge_type.shape}, \n if_multi_relational={self.if_multi_relational}, \n n_edge_type={self.n_edge_type})'

	def add_in_svertex(self, vertex_name: str):
		self.in_svertex_list.append(vertex_name)

	def add_out_svertex(self, vertex_name: str):
		self.out_svertex_list.append(vertex_name)

	def set_name(self, vertex_name: str):
		self.name = vertex_name


class superGraph(object):
	def __init__(self, svertex_dict: dict[str, superVertex], sedge_dict: dict[tuple[str, str], torch.Tensor], setting_dict: dict):
		self.svertex_dict = svertex_dict
		self.sedge_dict = sedge_dict
		self.setting_dict = setting_dict

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
			self.svertex_dict[n2].add_in_svertex(n1)
			self.svertex_dict[n1].add_out_svertex(n2)

		for n, sv in self.svertex_dict.items():
			sv.set_name(n)
			if sv.in_svertex_list:
				sv.if_start_svertex = False

node_feat = torch.randn(4, 20)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
edge_type = torch.tensor([0, 0, 1, 1])

sv1 = superVertex(node_feat, edge_index, edge_type)
sv2 = superVertex(node_feat, edge_index, edge_type)
sv3 = superVertex(node_feat, edge_index, edge_type)

class svertexSetting(object):
	def __init__(self, vertex_name: str, inter_feat_dim: int, inter_agg_dim: list[int], mod: str=None, num_bases: int=32, if_catout: bool=True) -> None:
		self.verqtex_name = vertex_name
		self.inter_feat_dim = inter_feat_dim
		self.inter_agg_dim = inter_agg_dim
		self.mod = mod
		self.num_bases = num_bases
		self.if_catout = if_catout

sv_setting = svertexSetting('start_svertex', 20, [10, 10])
sv_setting_task = svertexSetting('task_svertex', 20, [10, 10], 'add')

sg = superGraph(
	svertex_dict={'sv1': sv1, 'sv2': sv2, 'sv3': sv3},
	sedge_dict={
		('sv1', 'sv3'): torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]]), 
		('sv2', 'sv3'): torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])},
	setting_dict={
		'sv1': sv_setting, 
		'sv2': sv_setting, 
		'sv3': sv_setting_task}
)

class internalModule(torch.nn.Module):
	"""The internal module of a supervertex, which is composed of an internal feature layer and multiple internal aggregation layers."""

	def __init__(self, in_dim: int, n_edge_type: int, if_start_svertex: bool, setting: svertexSetting, exter_agg_dim: int=None) -> None:
		super(internalModule, self).__init__()
		# in and out dimension
		self.in_dim = in_dim
		self.out_dim = setting.inter_agg_dim[-1]

		self.n_edge_type = n_edge_type
		self.if_start_svertex = if_start_svertex
		self.setting = setting
		self.exter_agg_dim = exter_agg_dim

		self.__init_interal_feat_layer__()
		self.__init_interal_agg_layer__()

	def __init_interal_feat_layer__(self):
		"""internal feature layer"""

		self.embedding = torch.nn.Parameter(torch.Tensor(self.in_dim, self.setting.inter_feat_dim))
		self.embedding.requires_grad = True

		# reset parameters to be normally distributed
		self.embedding.data.normal_()	

	def __init_interal_agg_layer__(self):
		"""internal aggregation layers"""

		# compute the dim of input of the first internal aggregation layer
		self.in_agg_dim = self.setting.inter_feat_dim
		if not self.if_start_svertex:
			assert self.setting.mod in ['cat', 'add'], f'The mod {self.setting.mod} is not supported. Please use cat or add.'
			
			if self.setting.mod == 'cat':
				assert self.exter_agg_dim, 'The exter_agg_dim is not set.'
				self.in_agg_dim += self.exter_agg_dim
			else:
				assert self.in_agg_dim == self.exter_agg_dim, 'The in_agg_dim is not equal to exter_agg_dim.'

		# create and initialize the internal aggregation layers
		self.n_internal_agg_layer = len(self.setting.inter_agg_dim)
		tmp_dim = [self.in_agg_dim] + self.setting.inter_agg_dim

		if self.n_edge_type == 1:
			# using GCN if there is only one edge type
			self.internal_agg_layers = torch.nn.ModuleList(
				[GripNetGCN(tmp_dim[i], tmp_dim[i + 1], cached=True) for i in range(self.n_internal_agg_layer)]
			)
		else:
			# using RGCN if there are multiple edge types
			assert self.n_edge_type > 1
			after_relu = [False if i == 0 else True for i in range(self.n_internal_agg_layer)]
			self.internal_agg_layers = torch.nn.ModuleList(
				[GripNetRGCN(tmp_dim[i], tmp_dim[i + 1], self.n_edge_type, self.setting.num_bases, after_relu[i]) for i in range(self.n_internal_agg_layer)]
			)


	def forward(self, x, edge_index, edge_type, edge_weight, range_list=None, if_catout=True) -> torch.Tensor:
		pass

sv = sg.svertex_dict['sv1']
svs = sg.setting_dict['sv1']

inter_model = internalModule(sv.n_node_feat, sv.n_edge_type, sv.if_start_svertex, svs)

# %%
