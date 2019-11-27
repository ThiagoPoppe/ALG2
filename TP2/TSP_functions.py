import numpy as np
import networkx as nx

from math import ceil
from queue import PriorityQueue

class Node:
    """
        Classe para representar um nó na árvore do Branch and Bound
    """
    
    def __init__(self, bound, level, cost, sol):
        self.bound = bound
        self.level = level
        self.cost = cost
        self.sol = sol
    
    # Comparação entre Nodes
    def __lt__(self, other):
        return self.bound < other.bound
    
    # Apenas para fins de debug
    def __str__(self):
        return 'bound: {}, level: {}, cost: {}, sol: {}'.format(self.bound, self.level, self.cost, self.sol)

def generate_points(lower=0, upper=100, size=4):
    """
        Função que gera pontos com coordenadas inteiras em R²
        
        Parâmetros:
        ----------
        lower : int (opcional)
            O menor valor que pode ser gerado pelo gerador (por padrão é 0)
            
        upper : int (opcional)
            O maior valor que pode ser gerado pelo gerador (por padrão é 10)
            
        size : int (opcional)
            Número de pontos a serem gerados. Iremos gerar 2^size pontos, onde size
            deve ser um valor no intervalo [4, 10] (por padrão é 4)
            
        Retorno:
        -------
        Caso o valor de size seja inviável, a função retornará None. Senão, retornará uma
        lista de pontos com coordenadas inteiras em R²   
    """
    
    # Verificando se o parâmetro size é viável
    if size < 4 or size > 10:
        print('*** O parâmetro size deve ter valor entre [4, 10] ***')
        return None
    
    # Gerando 2**size pontos inteiros
    points = []
    for i in range(2**size):
        p = np.random.randint(lower, upper, 2)
        points.append(p)
        
    return points

def get_euclidian_distance(points):
    """
        Função que calcula a distância euclidiana entre todos os pontos
        
        Parâmetros:
        ----------
        points : list of numpy.ndarray
            Lista contendo os pontos no plano R²
            
        Retorno:
        -------
        A função retorna uma lista de tuplas, onde o primeiro e segundo elementos são
        os vértices e o terceiro a distância calculada entre os mesmos usando a distância
        euclidiana
    """
    
    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            edges.append((i, j, dist))
            
    return edges

def get_manhattan_distance(points):
    """
        Função que calcula a distância manhattan entre todos os pontos
        
        Parâmetros:
        ----------
        points : list of numpy.ndarray
            Lista contendo os pontos no plano R²
            
        Retorno:
        -------
        A função retorna uma lista de tuplas, onde o primeiro e segundo elementos são
        os vértices e o terceiro a distância calculada entre os mesmos usando a distância
        manhattan
    """
    
    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.abs(points[i] - points[j]).sum()
            edges.append((i, j, dist))
            
    return edges

def bound(G, nodes):
    """
        Função para computar o bound para o algoritmo Branch and Bound
        
        Parâmetros:
        ----------
        G : grafo
            Grafo de entrada para o problema
            
        edges : list of tuple
            Lista de arestas que devem ter na nossa solução atual
            
        Retorno:
        -------
            Retorna o valor de bound para um nó na árvore do Branch and Bound  
    """
    
    # Convertendo a lista de vértices em listas de arestas
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    
    # Computando os pesos que devem ter na solução atual
    estimative = 0
    for (u, v) in edges:
        estimative += (2 * G[u][v]['weight'])
    
    for u in G.nodes():          
        # Encontrando as arestas que devem existir incidentes à 'u' 
        u_edges = list(filter(lambda x: x[0] == u or x[1] == u, edges))
        
        # Se tivermos 2 arestas podemos continuar para o próximo vértice
        if len(u_edges) < 2:
            # Computando os dois menores pesos das arestas que incidem àquele vértice
            u_data = list(G.edges(u, data=True))
            u_data = sorted(u_data, key=lambda x: x[2]['weight'])
            w1, w2 = u_data[0][2]['weight'], u_data[1][2]['weight']
            
            # Caso não tenhamos nenhuma aresta necessária, podemos somar os dois pesos
            if len(u_edges) == 0:
                estimative += w1
                estimative += w2
            
            # Senão somamos um dos dois pesos (não somando o mesmo peso da aresta duas vezes)
            elif len(u_edges) == 1:
                u, v = u_edges[0]
                must_have_weight = G[u][v]['weight']
                estimative += w1 if w1 < must_have_weight else w2
    
    return ceil(estimative / 2)

def branch_and_bound(graph):
    """ 
        Função que implementa o algoritmo Branch and Bound para o problema do TSP

        Parâmetros:
        ----------
        graph : grafo
            Grafo de entrada para o problema

        Retorno:
        -------
        A função retorna a ordem dos vértices do melhor caminho e o seu tamanho
    """

    # Copiando o grafo para evitar possíveis modificações não desejadas
    G = graph.copy()

    # Criando o primeiro nó da árvore do Branch and Bound
    root = Node(bound(G, []), 1, 0, [0])
    heap = PriorityQueue()
    heap.put(root)
    
    # Definindo o melhor como infinito e a solução como vazia
    best = np.inf
    solution = []
    
    # Percorrendo o min-heap
    while not heap.empty():
        node = heap.get()

        # Caso cheguemos em uma folha, iremos verificar se a mesma é
        # melhor do que uma solução que já achamos
        if node.level == G.number_of_nodes():
            if node.cost < best:
                best = node.cost
                solution = node.sol

        elif node.bound < best:
            if node.level < G.number_of_nodes():
                for k in range(G.number_of_nodes()):
                    v = node.sol[-1]
                    # Aprofundando o Node se 'k' não estiver na solução, existir a aresta (v, k) e bound (com 'k') melhor que o best
                    if k not in node.sol and G.has_edge(v, k) and bound(G, node.sol + [k]) < best:
                        new_sol = node.sol + [k]
                        new_cost = node.cost + G[v][k]['weight']
                        new_bound = bound(G, new_sol)
                        heap.put(Node(new_bound, node.level+1, new_cost, new_sol))
                        
    # Adicionando o vértice inicial para fechar o circuito
    solution.append(0)
    
    # Computando o tamanho do caminho encontrado
    length = 0
    for i in range(len(solution)-1):
        u, v = solution[i], solution[i+1]
        length += G[u][v]['weight']
        
    return solution, length

def twice_around_the_tree(graph):
    """
        Função que implementa o algoritmo 2-aproximativo para o problema do TSP
        
        Parâmetros:
        ----------
        graph : grafo
            Grafo de entrada para o problema
            
        Retorno:
        -------
        A função retorna a ordem dos vértices do caminho aproximado e o seu tamanho
    """

    # Copiando o grafo para evitar possíveis modificações não desejadas
    G = graph.copy()
    
    # Encontrando a árvore geradora mínima do grafo
    MST = nx.minimum_spanning_tree(G)
    
    # Caminhando em pré-ordem pela árvore usando o vértice 0 como raiz e fechando o ciclo
    # Hamiltoniano, conectando o vértice final ao inicial
    walk = list(nx.dfs_preorder_nodes(MST, source=0))
    walk.append(0)

    # Computando o tamanho do caminho encontrado
    length = 0
    for i in range(len(walk)-1):
        u, v = walk[i], walk[i+1]
        length += G[u][v]['weight']
    
    return walk, length    

def christofides(graph):
    """
        Função que implementa o algoritmo 1.5-aproximativo para o problema do TSP
        
        Parâmetros:
        ----------
        graph : grafo
            Grafo de entrada para o problema
            
        Retorno:
        -------
        A função retorna a ordem dos vértices do caminho aproximado e o seu tamanho
    """
    
    # Copiando o grafo para evitar possíveis modificações não desejadas
    G = graph.copy()

    # Encontrando a árvore geradora mínima do grafo
    MST = nx.minimum_spanning_tree(G)
    
    # Criando o conjunto de vértices que possuem grau ímpar e montando um subgrafo induzido a partir dos mesmos
    odd_degree_nodes = []
    for node in MST.nodes():
        if MST.degree(node) % 2 == 1:
            odd_degree_nodes.append(node)
    
    induced_graph = G.subgraph(odd_degree_nodes)
    
    # Processando as arestas do grafo para achar o matching perfeito de peso mínimo
    
    u, v = list(induced_graph.edges())[0]
    max_weight = induced_graph[u][v]['weight']
    for (u, v) in induced_graph.edges():
        w = induced_graph[u][v]['weight']
        max_weight = w if w > max_weight else max_weight
        
    for (u, v) in induced_graph.edges():
        induced_graph[u][v]['weight'] = max_weight - induced_graph[u][v]['weight']
        
    # Encontrando o matching perfeito de peso mínimo e voltando os pesos para o original
    min_weight_matching = nx.max_weight_matching(induced_graph, maxcardinality=True)
    
    # Criando um subgrafo induzido com as arestas do matching perfeito de peso mínimo
    min_weight_matching_graph = G.edge_subgraph(min_weight_matching)
        
    # Criando um multigrafo com os vértices de G e arestas da MST e do matching perfeito de peso mínimo
    multigraph = nx.MultiGraph()
    multigraph.add_weighted_edges_from(MST.edges.data('weight'))
    multigraph.add_weighted_edges_from(min_weight_matching_graph.edges.data('weight'))
    
    # Computando o circuito euleriano
    eulerian_circuit = [u for (u, v) in nx.eulerian_circuit(multigraph, source=0)]
    
    # Retirando vértices repetidos, construindo assim um circuito hamiltoniano
    walk = []
    for node in eulerian_circuit:
        if node not in walk:
            walk.append(node)
    
    walk.append(0)
    
    # Computando o tamanho do caminho encontrado
    length = 0
    for i in range(len(walk)-1):
        u, v = walk[i], walk[i+1]
        length += graph[u][v]['weight']
        
    return walk, length