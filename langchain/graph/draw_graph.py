from IPython.display import Image, display
import matplotlib.pyplot as plt
from PIL import Image

from graph_chatbot import *


# jupyter 交互式环境查看图片
def draw_graph(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


# IDEA 非交互式环境查看图片
def draw_graph2(graph):
    with open('my_graph.png', 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
    # 加载图片
    img = Image.open('my_graph.png')
    # 使用matplotlib显示图片
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


draw_graph2(graph)
