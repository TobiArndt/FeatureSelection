import pydot
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np
def visualize_tree(root):
  graph = pydot.Dot(graph_type='digraph')
  def rek_call(graph, node):
    for n_k in node.children_nodes.keys():
      
      n = node.children_nodes[n_k]
      #print(node.key)
      #print(n.key)
      #print(10 * '*')
      graph.add_edge(pydot.Edge(str(node), str(n)))
      rek_call(graph, n)
  
  rek_call(graph, root)

  png_str = graph.create_png(prog='dot')
  sio = io.BytesIO()
  sio.write(png_str)
  sio.seek(0)
  file_bytes = np.asarray(bytearray(sio.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  img = imutils.resize(img, width = img.shape[1] * 2)
  cv2.imwrite('Experiment_Tree.png', img)