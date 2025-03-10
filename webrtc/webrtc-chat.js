document.addEventListener('DOMContentLoaded', (event) => {
    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('messageForm');
    const messageInput = document.getElementById('messageInput');
    const startButton = document.getElementById('startButton');

    let peerConnection;
    let dataChannel;
    let isOffer

    // Signaling server setup (using WebSocket as an example)
    const signalingServerUrl = 'ws://localhost:8080';
    const signalingSocket = new WebSocket(signalingServerUrl);

    // 创建 PeerConnection 和 DataChannel
    function createPeerConnection() {
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' } // 使用 Google 的 STUN 服务器
            ]
        };

        peerConnection = new RTCPeerConnection(configuration);

        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                signalingSocket.send(JSON.stringify({ candidate: event.candidate }));
            }
        };
        if(isOffer){
            dataChannel = peerConnection.createDataChannel('chat');
            dataChannel.onopen = () => {
                console.log('Offer Data channel is open');
            };
            dataChannel.onmessage = (event) => {
                console.log('Received message:', event.data); // 打印接收到的消息
                const messageElement = document.createElement('div');
                messageElement.textContent = `Received: ${event.data}`;
                messagesDiv.appendChild(messageElement);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            };
        }else{
            peerConnection.ondatachannel = (event) => {
            console.log('Received data channel from Offer');
            dataChannel = event.channel;

            // 绑定 onmessage 事件处理程序
            dataChannel.onmessage = (event) => {
                console.log('Received message:', event.data); // 打印接收到的消息
                const messageElement = document.createElement('div');
                messageElement.textContent = `Received: ${event.data}`;
                messagesDiv.appendChild(messageElement);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            };

            // 监听 dataChannel 状态变化
            dataChannel.onopen = () => {
                console.log('Data channel is open');
            };

            dataChannel.onclose = () => {
                console.log('Data channel is closed');
            };
        };
        }

    }

    // 处理信令消息
    function handleSignalingMessage(data) {
        if (data.sdp) {
            if (data.sdp.type === 'offer') {
                // 如果是 offer 消息，接收端需要创建 peerConnection 并设置远程描述
                if (!peerConnection) {
                    createPeerConnection();
                }
                peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp))
                    .then(() => {
                        // 接收端创建 answer 并发送回发起方
                        return peerConnection.createAnswer().then(createDescription).catch(handleError);
                    }).catch(handleError);
            } else if (data.sdp.type === 'answer') {
                // 如果是 answer 消息，发起端设置远程描述
                if (peerConnection) {
                    peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp)).catch(handleError);
                } else {
                    console.error('peerConnection is not initialized for answer');
                }
            }
        } else if (data.candidate) {
            if (peerConnection) {
                peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(handleError);
            } else {
                console.error('peerConnection is not initialized for candidate');
            }
        }
    }

    // 发起呼叫
    function call() {
        if (!peerConnection) {
            createPeerConnection();
            peerConnection.createOffer()
                .then(createDescription)
                .catch(handleError);
        } else {
            console.warn('Peer connection already exists');
        }
    }

    // 创建并设置本地描述
    function createDescription(description) {
        return peerConnection.setLocalDescription(description).then(() => {
            signalingSocket.send(JSON.stringify({ sdp: peerConnection.localDescription }));
        }).catch(handleError);
    }

    function handleError(error) {
        console.error('Error occurred', error);
    }

    // 发送消息
    messageForm.onsubmit = (event) => {
        event.preventDefault();
        const message = messageInput.value;
        if (message && dataChannel.readyState === 'open') {
            dataChannel.send(message);
            const messageElement = document.createElement('div');
            messageElement.textContent = `Sent: ${message}`;
            messagesDiv.appendChild(messageElement);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            messageInput.value = '';
        }
    };

    // 开始按钮
    if (startButton) {
        startButton.onclick = () => {
            isOffer = true;
            call();
        };
    } else {
        console.error('Start button not found');
    }

    // 监听信令消息
    signalingSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received signaling message:', data);
        handleSignalingMessage(data);
    };

    // 处理 WebSocket 连接错误
    signalingSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    // 处理 WebSocket 连接关闭
    signalingSocket.onclose = () => {
        console.log('WebSocket connection closed');
    };
});