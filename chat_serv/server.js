// MCP Chat Server for LLM Clients
const WebSocket = require('ws');
const crypto = require('crypto');
const http = require('http');

// Simple in-memory database (would use a real DB in production)
const users = new Map(); // username -> {passwordHash, token}
const rooms = new Map(); // roomName -> {password, messages, subscribers}
const connections = new Map(); // token -> {ws, username, rooms}

// Create HTTP server
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('MCP Chat Server Running');
});

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Helper functions
function hashPassword(password) {
  return crypto.createHash('sha256').update(password).digest('hex');
}

function generateToken() {
  return crypto.randomBytes(32).toString('hex');
}

function sendToClient(ws, type, data) {
  ws.send(JSON.stringify({ type, data }));
}

function broadcastToRoom(roomName, type, data, excludeUsername = null) {
  const room = rooms.get(roomName);
  if (!room) return;

  room.subscribers.forEach(username => {
    if (username === excludeUsername) return;

    // Find all connections for this user
    for (const [token, conn] of connections.entries()) {
      if (conn.username === username) {
        sendToClient(conn.ws, type, {
          room: roomName,
          ...data
        });
      }
    }
  });
}

// WebSocket connection handler
wss.on('connection', (ws) => {
  let userToken = null;

  ws.on('message', (message) => {
    try {
      const { type, data } = JSON.parse(message);

      switch (type) {
        case 'register':
          // Register a new user
          const { username, password } = data;
          if (users.has(username)) {
            sendToClient(ws, 'error', { message: 'Username already exists' });
            return;
          }

          const passwordHash = hashPassword(password);
          users.set(username, { passwordHash, tokens: [] });
          sendToClient(ws, 'registered', { username });
          break;

        case 'login':
          // Login an existing user
          const { username: loginUser, password: loginPass } = data;
          const user = users.get(loginUser);

          if (!user || user.passwordHash !== hashPassword(loginPass)) {
            sendToClient(ws, 'error', { message: 'Invalid credentials' });
            return;
          }

          userToken = generateToken();
          user.tokens.push(userToken);

          connections.set(userToken, { ws, username: loginUser, rooms: new Set() });
          sendToClient(ws, 'loggedIn', { token: userToken });
          break;

        case 'createRoom':
          // Create a new chat room
          if (!userToken) {
            sendToClient(ws, 'error', { message: 'Not authenticated' });
            return;
          }

          const { roomName, roomPassword } = data;
          if (rooms.has(roomName)) {
            sendToClient(ws, 'error', { message: 'Room already exists' });
            return;
          }

          rooms.set(roomName, {
            password: hashPassword(roomPassword),
            messages: [],
            subscribers: new Set()
          });

          sendToClient(ws, 'roomCreated', { roomName });
          break;

        case 'joinRoom':
          // Join an existing room
          if (!userToken) {
            sendToClient(ws, 'error', { message: 'Not authenticated' });
            return;
          }

          const conn = connections.get(userToken);
          const { roomName: joinRoom, password: joinPassword } = data;
          const room = rooms.get(joinRoom);

          if (!room) {
            sendToClient(ws, 'error', { message: 'Room does not exist' });
            return;
          }

          if (room.password !== hashPassword(joinPassword)) {
            sendToClient(ws, 'error', { message: 'Invalid room password' });
            return;
          }

          room.subscribers.add(conn.username);
          conn.rooms.add(joinRoom);

          // Send room history to the client
          sendToClient(ws, 'joinedRoom', {
            roomName: joinRoom,
            history: room.messages.slice(-50) // Last 50 messages
          });

          // Notify others that someone joined
          broadcastToRoom(joinRoom, 'userJoined', { username: conn.username }, conn.username);
          break;

        case 'leaveRoom':
          // Leave a room
          if (!userToken) {
            sendToClient(ws, 'error', { message: 'Not authenticated' });
            return;
          }

          const leaveConn = connections.get(userToken);
          const { roomName: leaveRoom } = data;
          const leaveRoomData = rooms.get(leaveRoom);

          if (!leaveRoomData || !leaveConn.rooms.has(leaveRoom)) {
            sendToClient(ws, 'error', { message: 'Not in this room' });
            return;
          }

          leaveRoomData.subscribers.delete(leaveConn.username);
          leaveConn.rooms.delete(leaveRoom);

          sendToClient(ws, 'leftRoom', { roomName: leaveRoom });
          broadcastToRoom(leaveRoom, 'userLeft', { username: leaveConn.username });
          break;

        case 'sendMessage':
          // Send a message to a room
          if (!userToken) {
            sendToClient(ws, 'error', { message: 'Not authenticated' });
            return;
          }

          const msgConn = connections.get(userToken);
          const { roomName: msgRoom, content } = data;
          const msgRoomData = rooms.get(msgRoom);

          if (!msgRoomData || !msgConn.rooms.has(msgRoom)) {
            sendToClient(ws, 'error', { message: 'Not in this room' });
            return;
          }

          const timestamp = Date.now();
          const messageObj = {
            id: crypto.randomBytes(16).toString('hex'),
            username: msgConn.username,
            content,
            timestamp
          };

          msgRoomData.messages.push(messageObj);

          // Broadcast to all subscribers including sender (for confirmation)
          broadcastToRoom(msgRoom, 'newMessage', messageObj);
          break;

        default:
          sendToClient(ws, 'error', { message: 'Unknown command' });
      }
    } catch (error) {
      sendToClient(ws, 'error', { message: 'Invalid request format' });
    }
  });

  ws.on('close', () => {
    if (!userToken) return;

    const conn = connections.get(userToken);
    if (!conn) return;

    // Leave all rooms
    conn.rooms.forEach(roomName => {
      const room = rooms.get(roomName);
      if (room) {
        room.subscribers.delete(conn.username);
        broadcastToRoom(roomName, 'userLeft', { username: conn.username });
      }
    });

    // Remove the connection
    connections.delete(userToken);

    // Remove token from user's tokens list
    const user = users.get(conn.username);
    if (user) {
      user.tokens = user.tokens.filter(t => t !== userToken);
    }
  });
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`MCP Chat Server running on port ${PORT}`);
});