require('dotenv').config();
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const apiRoutes = require('./routes/api');

const app = express();
app.use(cors());
app.use(express.json({ limit: '200kb' }));

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/healthbot';
const PORT = parseInt(process.env.PORT || '5000', 10);

mongoose.connect(MONGODB_URI).catch((err) => {
  console.error('MongoDB connection error', err);
});
const db = mongoose.connection;
db.on('error', (err) => console.error('MongoDB error', err));
db.once('open', () => console.log('MongoDB connected'));

const interactionSchema = new mongoose.Schema({
  userId: { type: String, default: 'demo' },
  message: String,
  response: String,
  severity: Number,
  createdAt: { type: Date, default: Date.now }
});
const Interaction = mongoose.model('Interaction', interactionSchema);

app.use((req, _res, next) => {
  req.models = { Interaction };
  req.reqId = Math.random().toString(36).slice(2);
  next();
});

app.use('/api', apiRoutes);

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`healthbot backend running on port ${PORT}`);
  });
}

module.exports = app;
