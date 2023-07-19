import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from matplotlib import colors as mcolors
import copy
import random
import numpy as np
import math 



def parse_point(point_str):
    # Extract x, y, z values from the string
    match = re.search(r'\{(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)\}', point_str)
    if match:
        return tuple(float(x) for x in match.groups()[::2]) # Extracting only the x, y, z values
    return None

def read_points_from_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            point = parse_point(line)
            if point:
                points.append(point)
    return points

def create_graph_from_files(start_points_file, end_points_file):
    start_points = read_points_from_file(start_points_file)
    end_points = read_points_from_file(end_points_file)

    if len(start_points) != len(end_points):
        raise ValueError('Start and end points files have different lengths.')

    graph = nx.Graph()
    for i, (start, end) in enumerate(zip(start_points, end_points)):
        graph.add_edge(start, end, index=i)

    return graph

def make_oieler(g):
    fixer = (100,100,100)
    g.add_node(fixer)
    for v in g.nodes():
        if (g.degree(v) % 2 == 1):
            g.add_edge(v,fixer)
    if(g.degree(fixer) == 0):
        g.remove_node(fixer)
    return g
            


# Plot the 3D graph
def plot_3d_graph(graph):
    fig = plt.figure(figsize=(20, 20), dpi=100)  # Adjust the figsize and dpi as needed
    ax = fig.add_subplot(111, projection='3d')

    xyz = {node: node for node in graph.nodes()}
    for edge in graph.edges():
        x, y, z = zip(*[xyz[edge[0]], xyz[edge[1]]])
        ax.plot(x, y, z, color='b', lw=1, alpha=0.6)

    for node, pos in xyz.items():
        ax.scatter(*pos, s=50, c='r', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def meshVertexGraph(mesh, weightMode):
    """ Make a networkx graph with mesh vertices as nodes and mesh edges as edges """

    # Create graph
    g = nx.Graph()

    #######
    edgeOrder = {}
    #######

    for i in range(mesh.Vertices.Count):
        # Get vertex as point3D
        pt3D = rc.Geometry.Point3d(mesh.Vertices.Item[i])

        # Add node to graph and get its neighbours
        g.add_node(i, point=pt3D)
        neighbours = mesh.Vertices.GetConnectedVertices(i)

        ########## Create orthogonal line to The beginning of labor
        p=rc.Geometry.Point3d(mesh.Vertices.Item[i].X - 1, mesh.Vertices.Item[i].Y, mesh.Vertices.Item[i].Z)
        order = {}
        ##########
        
        # Add edges to graph
        for n in neighbours:
            ######## Calculate angle between orthogonal line and edge
            line = rc.Geometry.Line(mesh.Vertices.Item[i], mesh.Vertices.Item[n])
            p1=rc.Geometry.Vector3d(mesh.Vertices.Item[i].X, mesh.Vertices.Item[i].Y, mesh.Vertices.Item[i].Z)
            p2=rc.Geometry.Vector3d(mesh.Vertices.Item[n].X, mesh.Vertices.Item[n].Y, mesh.Vertices.Item[n].Z)
            order[n] = rc.Geometry.Vector3d.VectorAngle(p1, p2)
            ########
            if n > i:
                line = rc.Geometry.Line(mesh.Vertices.Item[i], mesh.Vertices.Item[n])
                if weightMode == "edgeLength":
                    w = line.Length
                elif weightMode == "sameWeight":
                    w = 1
                elif weightMode == "edgeSlope":
                    w = lineSlope(line)
                g.add_edge(i, n, weight=w, line=line)
        #initial order and mark all edges as not visited
        order = sorted(order)
        edgeOrder[i] = [(j, False) for j in order]
    return [g, edgeOrder]


#####################

def find(G, orders,source = None):
    if not nx.is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian.")
    g = G.__class__(G)  # copy graph structure (not attributes)
    # set starting node
    if source is None:
        edges = g.edges
        v = random.choice(list(g.nodes()))
    else:
        v = source
    if g.is_directed():
        degree = g.in_degree
        edges = g.in_edges
        get_vertex = itemgetter(0)
    else:
        degree = g.degree
        edges = g.edges
        get_vertex = itemgetter(1)
    vertex_stack = [v]
    first = v
    last_vertex = None
    last_ver = None
    ret = {}
    k = 0
    TrMkr = {}
    edge_map = {}
    for v in g.nodes():
        edge_map[v] = {}
    ret[k] = []
    dead_end = False
    newVer = True
    while vertex_stack:
        current_vertex = vertex_stack[-1]
       
        if degree(current_vertex) == 0 or dead_end:
            last_vertex = current_vertex
            last_ver = current_vertex
          
            vertex_stack.pop()
            dead_end = False
            if(not newVer):
                if(current_vertex not in TrMkr.keys()):
                    TrMkr[current_vertex] = []
            if(len(ret[k]) != 0):
                k += 1
                ret[k] = []
            newVer = True

        else:
            if(newVer):
                last_ver = None          
            arbitrary_edge = GetNextEdge(current_vertex, last_ver, orders,first) 
            if (arbitrary_edge != None):
                vert = arbitrary_edge[0]
                if(newVer):
                    if(current_vertex not in TrMkr.keys()):
                        TrMkr[current_vertex] = []
                    newVer = False
                last_ver = current_vertex
                vertex_stack.append(vert)
               
                if (g.has_edge(vert,current_vertex) and vert != current_vertex):
                    ret[k].append((current_vertex,vert))
                    dd ={}
                    dd['key'] = k
                    #dd['in_route'] = ret[k].index((current_vertex,vert))
                    dd['clockwise'] =  orders[current_vertex].index((vert, False))
                    dd['In'] = False
                    edge_map[current_vertex][vert] = dd
                    ddd = {}
                    ddd['key'] = k
                    #ddd['in_route'] = ret[k].index((current_vertex,vert))
                    ddd['clockwise'] =  orders[vert].index((current_vertex, False))
                    ddd['In'] = True
                    edge_map[vert][current_vertex] = dd
                    g.remove_edge(vert, current_vertex)
                else:
                    g.remove_edge(current_vertex, vert)
                prevOrder = orders[current_vertex].index((vert, False))
                orders[current_vertex][prevOrder] = (vert, True)
                prevOrder = orders[vert].index((current_vertex, False))
                orders[vert][prevOrder] = (current_vertex, True)
            else:
                dead_end = True
            
    print("lenn "+str(g.edges()))
    return (ret,edge_map)

   


# for current vertex use GetBound to get list of possible edges to continue the circuit and choose one from them
def GetNextEdge(current_vertex, last_vertex, orders,first):
    if(last_vertex == None):
        
        return random.choice(GetBounds(current_vertex, 1, orders, -1)[0])
    else:
           
            prevOrder = orders[current_vertex].index((last_vertex, True))
            
    Lbound,Lbol = GetBounds(current_vertex, -1, orders, prevOrder)
    Rbound,Rbol = GetBounds(current_vertex, 1, orders, prevOrder)
    chose_list = Lbound + Rbound
    if(Lbol == False):
        chose_list = Rbound
    else:
        chose_list = Lbound      
    ##Sanity check
    if(Lbound == None or Rbound == None):
        return None
    if(len(chose_list) == 0):
        return None
    if(len(chose_list) == 3):
        chose_list.pop(1)
    a = random.choice(chose_list)
    if(len(chose_list) > 1 and orders[current_vertex][a][0] == first and FalseCounter(orders, chose_list, first) == 1):
        while(orders[current_vertex][a][0] == first):
            a = random.choice(Lbound + Rbound)
    return orders[current_vertex][a]

def FalseCounter(orders, chose_list, vertex):
    count = 0
    for i in range(len(orders[vertex])):
        if(orders[vertex][i][1] == False):
            count += 1
    return count

# for current vertex get list of possible edges to continue the circuit
def GetBounds(current_vertex, offset, orders, prevOrder):
    ret = []
    ## Last vertex not exist
    if(prevOrder == -1):
        for i in range(len(orders[current_vertex])):
            if(orders[current_vertex][i][1] == False):
                ret.append(orders[current_vertex][i])
        return [ret,True]
    i = (prevOrder + (1*offset)) % len(orders[current_vertex])
    j = 0
    while(orders[current_vertex][i][1] == False):
        if (i % 2 != prevOrder % 2):
            ret.append(i)
        j+= 1
        i = (i + 1*offset) % len(orders[current_vertex])

    bol = True
    if(j % 2 == 0):
        bol = False
    return [ret,bol]

def con(ret,orders):
    r = []
    for i in range(len(ret.keys())--1):
        r = ConnectLists(ret[i],ret[i+1],orders)
    return r

def ConnectLists(listA, listB,orders):
    if(len(listA) == 0):
        return listB
    if(len(listB) == 0):
        return listA
    connectNode = listA[0]
    ret = []
    i = 0
    j = len(listB) - 1
    while(listB[j] != ConnectNode):
        j -= 1
    while(listB[i] != connectNode):
        ret.append(listB[i])
        i+=1
    ret.append(listB[i])
    if(j != i and i != 0):
        for k in range(i,j):
            ret.append(listB[k])
        i = j - 1
    
        
        
    if(i == 0):
        prevv = listB[len(listB)-2]
    else:
        prevv = listB[i-1]
    if(j == len(listB) - 1):
        nextv = listB[1]
    else:
        nextv = listB[j+1]
    BForder = orders[connectNode].index((prevv, True))
    BLorder = orders[connectNode].index((nextv, True))
    AForder = orders[connectNode].index((listA[len(listA)-2],True))
    ALorder = orders[connectNode].index((listA[1],True))

    if(ALorder > AForder):
        k = 1
    else:
        k = -1
    kk = Aorder
    while(kk != BForder or kk != BLorder):
        kk += k % len(orders[connectNode])
    bol
    if(kk == BForder):
        Bol = 1
        mov = BForder
    else: 
        Bol = -1
        mov = BLorder
    k = mov + Bol
    while(k != mov):
        ret.append(listA[k])
        k += Bol
    for k in range(i,len(listB)):
        ret.append(listB[k])
    return ret 





def initTestG(g):
    edgeOrder = {}
    for i in g.nodes():
        order = {}
        relative_line = []
        order = sort_points_clockwise(np.array(list(g.neighbors(i))))

        edgeOrder[i] = [((j[0],j[1],j[2]), False) for j in order]

    return edgeOrder

def centroid(points):
    return np.mean(points, axis=0)

def project_to_2d(points, plane_normal):
    points = np.array(points)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    return points - points.dot(plane_normal)[:, np.newaxis] * plane_normal

def polar_angle(point, reference):
    diff = point - reference
    
    angle = np.arctan2(diff[1].real, diff[0].real)
   
   
    return angle if angle >= 0 else 2 * np.pi + angle

def compute_plane_normal(points):
    points_centered = points - centroid(points)
    cov_matrix = np.dot(points_centered.T, points_centered)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    return eig_vecs[:, np.argmin(eig_vals)]

def sort_points_clockwise(points):
    points = np.array(points)
    plane_normal = compute_plane_normal(points)
    points_2d = project_to_2d(points, plane_normal)
    center = centroid(points_2d)

    sorted_indices = sorted(range(len(points)), key=lambda i: polar_angle(points_2d[i, :2], center[:2]))
    return [points[i] for i in sorted_indices]

def fill_Trouble_Maker(TrMkr,orders,routD):
    for node in TrMkr.keys():
        for k in routD.keys():
            index_lst = []
            route = routD[k]
            for i in range(len(route)):
                if(route[i][1] == node):
                    TrMkr[node].append((route[i][0],k,True,orders[node].index((route[i][0], True))))
                    if(route[(i+1) % len(route)][0] == node):
                        TrMkr[node].append((route[(i+1) % len(route)][1],k,False,orders[node].index((route[(i+1) % len(route)][1], True))))
            TrMkr[node] = sorted(TrMkr[node], key=lambda x: x[3])
   
    return TrMkr


def fixOrder(lst,routD):
    bol = lst[0][2]
  
    for i in range(2,len(lst),2):
        if(lst[i][1] != lst[0][1] and bol != lst[i][2]):
            routD[lst[i][1]] = switch_route(routD[lst[i][1]])
            lst[i] = set_tuple(lst[i],2)
            for j in range(1,len(lst),1):
                if(lst[j][1] == lst[i][1]):
                    lst[j] = set_tuple(lst[j],2)
                    
                    
def index_list(rout,index,u,v,b = False):
    lst = rout[index]
    for i in range(len(lst)):
        if(lst[i] == (u,v)):
            return i
    rout[index] = switch_route(lst)
    if(b):
        print("----------------")
        print(rout)
        print(index)
        print(str(v) + " , "+str(u))
        print("-----------------")
        return None
    return index_list(rout,index,u,v,True)
    
                    
def concat_routs(A,B,anode,bnode,node,routD):
    ret = []
    j = -1
    print(str(A)+" do connect with "+str(B))
    print(anode)
    print(node)
    print(bnode)
    
    print("FirstRout")
    print(routD[A])
    
    print("SecRout")
    print(routD[B])
    
    print(" ")
    
    a = index_list(routD,A,anode,node)
    b = index_list(routD,B,node,bnode)
    print("a "+str(a)+ " b "+str(b))

    ret = routD[A][:((a+1) % len(routD[A]))]+routD[B][b:]+routD[B][:b]+routD[A][((a+1) % len(routD[A])):]
    
    print(ret)
    routD[A] = ret
    routD[B] = ret
    return routD
      


def remove_rout(lst,k):
    ret = []
    for tup in lst:
        if(tup[1] != k):
            ret.append(tup)
    return ret
            
    
def set_tuple(tup, k):
    # Convert the tuple to a list
    temp_list = list(tup)
    
    # Change the value at index k
    temp_list[k] = not temp_list[k]
    
    # Convert the list back to a tuple and return it
    return tuple(temp_list)    


    
def connect_routes(TrMkr,routD):
    visited = [[True if i == j else False for j in routD.keys()] for i in routD.keys()]
    
    
    print(TrMkr)
    for node in TrMkr.keys():
        fixOrder(TrMkr[node],routD)
        while(not same(TrMkr[node])):
            
            print(TrMkr[node])
            for i in range(len(TrMkr[node])-1):
                if(TrMkr[node][i][2] == True and TrMkr[node][(i+1) % len(TrMkr[node])][1] == TrMkr[node][(i+2) % len(TrMkr[node])][1]):
                    found = i
                    A = TrMkr[node][i][1]
                    B = TrMkr[node][(i+1) % len(TrMkr[node])][1]
                    if(not visited[A][B]):
                        routD = concat_routs(A,B,TrMkr[node][i][0],TrMkr[node][(i+1) % len(TrMkr[node])][0], node,routD)
                        TrMkr[node] = remove_rout(TrMkr[node],B)
                        visited[B][A] = True
                        visited[A][B] = True
                        i = len(TrMkr[node])
                    else:
                        TrMkr[node] = remove_rout(TrMkr[node],B)
                    break
                
            if(found == -1):
                print("swapp")
                switch_route(routD[TrMkr[node][0][1]])
                TrMkr[node][0] = set_tuple(TrMkr[node][0],2)
                for j in range(len(TrMkr[node])):
                    if(TrMkr[node][j][1] ==  TrMkr[node][0][1]):
                         TrMkr[node][j] = set_tuple(TrMkr[node][j],2)
                fixOrder(TrMkr[node],routD)
            
    return routD

#make sure g will be Eulearian by "adding" fixer node.               
def make_oieler(g):
    fixer = (100,100,100)
    g.add_node(fixer)
    for v in g.nodes():
        if (g.degree(v) % 2 == 1):
            g.add_edge(v,fixer)
            
    if(g.degree(fixer) == 0):
        g.remove_node(fixer) 
      
    return g

    
#Check whether returned route contatins every edge from g
def saniti_Check(g,lst):
    g_tag = nx.Graph()
    g_tag.add_nodes_from(list(g.nodes()))
    cnt = 0
    for i in range(len(lst)):
        g_tag.add_edge(lst[i][0],lst[i][1])
        if(lst[i][1] != lst[(i+1) % len(lst)][0]):
            cnt+= 1
            print("we need to talk")
            print(lst[i])
            print(lst[i+1])
    for e in lst:
        g_tag.add_edge(e[0],e[1])
    for e in g.edges():
        if(not g_tag.has_edge(e[0],e[1]) and not g_tag.has_edge(e[1],e[0])):
            print("oh no.. what have you done??? "+ str(e[0])+" , "+str(e[1]))
            cnt += 1
    if(cnt == 0):
        print("you are the mann")
        return True
    return False
        

def plot_3d_dict(graph_dict, figsize=(20, 20), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the list of color names
    colors = list(mcolors.CSS4_COLORS.keys())
    
    for i, edges in enumerate(graph_dict.values()):
        # If there are more keys than colors, we loop back to the start of the colors list
        color = colors[i % len(colors)]
        
        for edge in edges:
            # Assuming the edge is a tuple of points, each of which is a tuple of coordinates
            x_coords = [point[0] for point in edge]
            y_coords = [point[1] for point in edge]
            z_coords = [point[2] for point in edge]
            
            ax.plot(x_coords, y_coords, z_coords, color=color, lw=1, alpha=0.6)
            ax.scatter(x_coords, y_coords, z_coords, s=50, c=color, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
import ast


    
def reverse_dict(d):
    return {v: k for k, v in d.items()}


def replace_values_with_dict_values(d, order):
    # reverse the dictionary d to make the lookup process easier
    o = {}
    # iterate over the keys in the order dictionary and replace the values
    for key in order.keys():
        new_values = [(d[val[0]], val[1]) if val[0] in d else val for val in order[key]]
        o[d[key]] = new_values

    return o

def check_circles(d):
    for i in d.keys():
        if(len(d[i]) > 0 and d[i][0][0] != d[i][len(d[i])-1][1]):
            print("----------------------Eror---------------------")
            return False
    return True

# Return the UnionFind structure and the "trouble nodes" dictionary.
#"touble nodes" are nodes that are contained in more than one route.
def init_phase_two(routes,data):
    met = nx.utils.UnionFind()
    trouble_nodes = {}
    for k in routes.keys():
        if(len(routes[k]) > 0):
            met[k]
            if(routes[k][0][0] not in trouble_nodes.keys()):
                trouble_nodes[routes[k][0][0]] = data[routes[k][0][0]]
    return (met,trouble_nodes)

def sort_by_clockwise(dictionary):
    return sorted(dictionary.keys(), key=lambda k: dictionary[k]['clockwise'])


# Verify if the paths of the nodes are connected to the same primary path
def same(met,nodata,node):
    bol = True
    k = None
    for v in nodata[node].values():
        if(bol == True):
            k = v['key']
            bol = False
        else:
            if(met[v['key']] != met[k]):
                return False
    return True

# Connect the semi-paths into a single, non-intersecting Eulerian path
def connect_dict(routes,met,nodata):
    OGroutes = copy.deepcopy(routes)
    for node in nodata.keys():
        order = sort_by_clockwise(nodata[node])
        i = 0
        ln = len(order)
        while(not same(met,nodata,node)):
            k1 = met[nodata[node][order[i]]['key']]
            k2 = met[nodata[node][order[(i+1) % ln]]['key']]
            v1 = order[i]
            v2 = order[(i+1)% ln]
            if(k1 != k2):
                i1,bol1 = get_index(routes[k1],v1,node)
                i2,bol2 = get_index(routes[k2],v2,node)
                if(bol1):
                    if(bol2):
                        routes[k2] = switch_route(routes[k2])
                        i2,bol2 =  get_index(routes[k2],v2,node)
                    routes[k1] = connect_routes(routes,nodata,node,v1,v2,k1,k2,i1,i2)
                    routes[k2] = routes[k1]
                else:
                    if(not bol2):
                        routes[k2] = switch_route(routes[k2])
                        i2,bol2 =  get_index(routes[k2],v2,node)  
                    routes[k1] = connect_routes(routes,nodata,node,v2,v1,k2,k1,i2,i1)
                    routes[k2] = routes[k1]
                met.union(nodata[node][order[i]]['key'],met[nodata[node][order[(i+1) % ln]]['key']])
                        
            i+= 1 %ln
    return routes[met[0]] 

#Find edge (v,node) in route
def get_index(route,v,node):
    for i in range(len(route)):
        if(route[i][0] == v and route[i][1] == node):
            return (i,True)
        if(route[i][0] == node and route[i][1] == v):
            return (i,False)
    return None

#switch route order
def switch_route(lst):
    ret = []
    for e in reversed(lst):
        ret.append((e[1],e[0]))
    return ret

            
            
def connect_routes(routes,nodata,node,v,u,kv,ku,iv,iu):
    return routes[kv][:iv+1]+routes[ku][iu:]+routes[ku][:iu]+routes[kv][iv+1:]

def generate_nonintersect_eulerian_path(start_points_file, end_points_file):
    # Create a 3D representation of the graph using networkx
    g = create_graph_from_files(start_points_file, end_points_file)

    # Initialize graph data. Create a dictionary to track the relative position of each edge with respect to nodes
    order = initTestG(g)

    # The 'find' function applies the foundational principles of Eulerian paths to create a list of "valid"
    # non-intersecting Eulerian semi-paths, which are stored in a dictionary
    ec = find(g, order)

    # Validate the results of the 'find' function
    bol = check_circles(ec[0])

    # If validation fails, return None
    if (not bol):
        return None

    # Initialize important data related to the process of connecting the semi-paths into one non-intersecting Eulerian path
    met,nodata = init_phase_two(ec[0],ec[1])

    # Connect the semi-paths into a single, non-intersecting Eulerian path
    ret = connect_dict(routes,met,nodata)
    #Check whether returned route contatins every edge from g
    if(saniti_Check(g,ret))
    return ret




def main():
    parser = argparse.ArgumentParser(description='Generate a non-intersecting Eulerian path for a 3D graph')
    parser.add_argument('start_points_file', type=str, help='File containing start points')
    parser.add_argument('end_points_file', type=str, help='File containing end points')

    args = parser.parse_args()

    result = generate_nonintersect_eulerian_path(args.start_points_file, args.end_points_file)

    return result

if __name__ == "__main__":
    main()


  
            
   


