const config = require('../config');
const crypto = require('crypto');

const TICK_INTERVAL = 500;
const STATS_BROADCAST_INTERVAL = 3000;

class TrafficSimulator {
  constructor() {
    this.intervalId = null;
    this.statsIntervalId = null;
    this.busy = false;
  }

  start(io, dataStore, mlService) {
    if (this.intervalId) return;

    console.log(`[Simulator] Starting traffic simulation every ${TICK_INTERVAL}ms`);

    this.intervalId = setInterval(() => {
      if (this.busy) return;
      this.busy = true;

      try {
        const startTime = process.hrtime.bigint();
        const features = this._generateFeatures();
        const result = mlService.fallbackPredict(features);
        const endTime = process.hrtime.bigint();
        const processingTime = Number(endTime - startTime) / 1e6;

        const event = {
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          features,
          prediction: result.prediction,
          confidence: result.confidence,
          probabilities: result.probabilities,
          processingTime
        };

        dataStore.addTraffic(event);
        io.emit('traffic-event', event);
      } catch (err) {
        console.error('[Simulator] Tick error:', err.message);
      } finally {
        this.busy = false;
      }
    }, TICK_INTERVAL);

    this.statsIntervalId = setInterval(() => {
      io.emit('stats-update', dataStore.getStats());
    }, STATS_BROADCAST_INTERVAL);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      clearInterval(this.statsIntervalId);
      this.intervalId = null;
      this.statsIntervalId = null;
      console.log(`[Simulator] Stopped traffic simulation`);
    }
  }

  _generateFeatures() {
    const isAttack = Math.random() < 0.3;
    let attackType = null;

    if (isAttack) {
      const types = ['ddos', 'port_scan', 'dns_spoof', 'mitm'];
      attackType = types[Math.floor(Math.random() * types.length)];
    }

    let features = {
      packet_size: Math.floor(64 + Math.random() * 1436),
      flow_duration: Math.random() * 50,
      packet_rate: Math.floor(10 + Math.random() * 990),
      byte_rate: Math.floor(Math.random() * 500000),
      protocol_type: Math.floor(Math.random() * 4),
      src_port: Math.floor(1024 + Math.random() * 64511),
      dst_port: Math.floor(Math.random() * 65535),
      flag_count: Math.floor(Math.random() * 5),
      iat_mean: 10 + Math.random() * 90,
      payload_entropy: 4.0 + Math.random() * 3.5
    };

    if (attackType === 'ddos') {
      features.packet_rate = 6000 + Math.random() * 4000;
      features.packet_size = Math.floor(64 + Math.random() * 64);
      features.iat_mean = Math.random() * 5;
    } else if (attackType === 'port_scan') {
      features.flag_count = 21 + Math.floor(Math.random() * 10);
      features.packet_size = 64;
      features.payload_entropy = 0.5 + Math.random() * 1.5;
    } else if (attackType === 'dns_spoof') {
      features.payload_entropy = Math.random() * 0.9;
      features.src_port = 53;
    } else if (attackType === 'mitm') {
      features.flow_duration = 101 + Math.random() * 100;
      features.packet_size = Math.floor(500 + Math.random() * 1000);
    }

    return features;
  }
}

module.exports = new TrafficSimulator();
