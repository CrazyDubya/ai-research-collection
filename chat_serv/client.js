// MCP Client for LLMs
const WebSocket = require('ws');
const readline = require('readline');

class MCPClient {
  constructor(serverUrl) {
    this.serverUrl = serverUrl;
    this.ws = null;
    this.token = null;
    this.username = null;
    this.currentRooms = new Set();
    this.messageHandlers = new Map();
    this.connected = false;
    this.connecting = false;

    // LLM context management
    this.llmContext = {
      conversationHistory: new Map(), // roomName -> array of messages
      contextSize: 10, // Number of messages to keep per room for context
    };
  }

  // Connect to the MCP server
  connect() {
    if (this.connected || this.connecting) return;

    this.connecting = true;
    this.ws = new WebSocket(this.serverUrl);

    this.ws.on('open', () => {
      this.connected = true;
      this.connecting = false;
      this._triggerHandler('connected');
      console.log('Connected to MCP server');
    });

    this.ws.on('message', (data) => {
      const message = JSON.parse(data);
      this._handleServerMessage(message);
    });

    this.ws.on('close', () => {
      this.connected = false;
      this._triggerHandler('disconnected');
      console.log('Disconnected from MCP server');
    });

    this.ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      this._triggerHandler('error', error);
    });
  }

  // Register a new user
  register(username, password) {
    this._sendToServer('register', { username, password });
  }

  // Login with credentials
  login(username, password) {
    this.username = username;
    this._sendToServer('login', { username, password });
  }

  // Create a new room
  createRoom(roomName, password) {
    this._sendToServer('createRoom', { roomName, password });
  }

  // Join an existing room
  joinRoom(roomName, password) {
    this._sendToServer('joinRoom', { roomName, password });
  }

  // Leave a room
  leaveRoom(roomName) {
    this._sendToServer('leaveRoom', { roomName });
    this.currentRooms.delete(roomName);
  }

  // Send a message to a room
  sendMessage(roomName, content) {
    if (!this.currentRooms.has(roomName)) {
      console.error(`Not joined to room: ${roomName}`);
      return;
    }

    this._sendToServer('sendMessage', { roomName, content });
  }

  // Get conversation context for the LLM
  getLLMContext(roomName) {
    if (!this.llmContext.conversationHistory.has(roomName)) {
      return [];
    }

    return this.llmContext.conversationHistory.get(roomName);
  }

  // Register event handlers
  on(event, handler) {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, []);
    }

    this.messageHandlers.get(event).push(handler);
  }

  // Internal: Send message to server
  _sendToServer(type, data) {
    if (!this.connected) {
      console.error('Not connected to server');
      return;
    }

    this.ws.send(JSON.stringify({ type, data }));
  }

  // Internal: Handle messages from server
  _handleServerMessage(message) {
    const { type, data } = message;

    // Special handling for authentication
    if (type === 'loggedIn') {
      this.token = data.token;
    }

    // Special handling for joining rooms
    if (type === 'joinedRoom') {
      this.currentRooms.add(data.roomName);

      // Initialize conversation history for this room
      if (!this.llmContext.conversationHistory.has(data.roomName)) {
        this.llmContext.conversationHistory.set(data.roomName, []);
      }

      // Add history messages to context
      const history = this.llmContext.conversationHistory.get(data.roomName);
      data.history.forEach(msg => {
        history.push({
          role: msg.username === this.username ? 'user' : 'other',
          username: msg.username,
          content: msg.content,
          timestamp: msg.timestamp
        });
      });

      // Trim to maintain context size
      if (history.length > this.llmContext.contextSize) {
        this.llmContext.conversationHistory.set(
          data.roomName,
          history.slice(history.length - this.llmContext.contextSize)
        );
      }
    }

    // Handle new messages and update LLM context
    if (type === 'newMessage') {
      const roomName = data.room;

      if (this.llmContext.conversationHistory.has(roomName)) {
        const history = this.llmContext.conversationHistory.get(roomName);

        history.push({
          role: data.username === this.username ? 'user' : 'other',
          username: data.username,
          content: data.content,
          timestamp: data.timestamp
        });

        // Trim to maintain context size
        if (history.length > this.llmContext.contextSize) {
          this.llmContext.conversationHistory.set(
            roomName,
            history.slice(history.length - this.llmContext.contextSize)
          );
        }
      }
    }

    // Trigger registered handlers
    this._triggerHandler(type, data);
  }

  // Internal: Trigger event handlers
  _triggerHandler(event, data) {
    if (this.messageHandlers.has(event)) {
      this.messageHandlers.get(event).forEach(handler => handler(data));
    }
  }

  // LLM integration: Process a message and generate response
  async processWithLLM(roomName, message, llmProcessor) {
    // Get conversation history for context
    const context = this.getLLMContext(roomName);

    // Call the LLM with context and new message
    const response = await llmProcessor(context, message);

    // Send the LLM response to the room
    this.sendMessage(roomName, response);

    return response;
  }
}

// Example integration with an LLM
async function exampleLLMProcessor(context, message) {
  // This would be replaced with an actual call to an LLM API
  console.log('Processing with LLM...');
  console.log('Context:', context);
  console.log('New message:', message);

  // Simulate LLM processing
  return `LLM response to: ${message}`;
}

// Simple CLI demo
if (require.main === module) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const client = new MCPClient('ws://localhost:3000');

  client.on('connected', () => {
    rl.question('Username: ', (username) => {
      rl.question('Password: ', (password) => {
        client.login(username, password);
      });
    });
  });

  client.on('loggedIn', (data) => {
    console.log(`Logged in with token: ${data.token}`);
    rl.question('Join room: ', (roomName) => {
      rl.question('Room password: ', (password) => {
        client.joinRoom(roomName, password);
      });
    });
  });

  client.on('joinedRoom', (data) => {
    console.log(`Joined room: ${data.roomName}`);
    console.log('Chat history:');
    data.history.forEach(msg => {
      console.log(`${msg.username}: ${msg.content}`);
    });

    rl.setPrompt('> ');
    rl.prompt();

    rl.on('line', async (line) => {
      if (line.startsWith('/quit')) {
        rl.close();
        process.exit(0);
      } else if (line.startsWith('/llm ')) {
        // Process with LLM
        const message = line.substring(5);
        const response = await client.processWithLLM(data.roomName, message, exampleLLMProcessor);
        console.log(`LLM responded: ${response}`);
      } else {
        client.sendMessage(data.roomName, line);
      }
      rl.prompt();
    });
  });

  client.on('newMessage', (data) => {
    if (data.username !== client.username) {
      console.log(`\n${data.username}: ${data.content}`);
      rl.prompt();
    }
  });

  client.on('error', (error) => {
    console.error('Error:', error.message);
    rl.prompt();
  });

  client.connect();
}

module.exports = MCPClient;