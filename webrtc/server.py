import asyncio
import json
import websockets

# 存储所有连接的客户端
connected_clients = set()

async def handle_message(websocket, path):
    # 将新连接的客户端添加到集合中
    connected_clients.add(websocket)
    print(f"New client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            # 解析收到的消息
            data = json.loads(message)
            print(f"Received message from {websocket.remote_address}: {data}")

            # 广播消息给所有其他连接的客户端
            for client in connected_clients:
                if client != websocket and client.open:
                    await client.send(json.dumps(data))
    except websockets.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        # 从集合中移除断开连接的客户端
        connected_clients.remove(websocket)

async def main():
    # 启动 WebSocket 服务器，监听 8080 端口
    async with websockets.serve(handle_message, "localhost", 8080):
        print("WebSocket signaling server is running on ws://localhost:8080")
        await asyncio.Future()  # 保持服务器运行

if __name__ == "__main__":
    asyncio.run(main())