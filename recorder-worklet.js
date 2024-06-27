class RecorderProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.recordings = [];
        this.isRecording = false;

        this.port.onmessage = (event) => {
            if (event.data.command === 'getRecording') {
                const audioData = this.getRecording();
                this.port.postMessage({ type: 'recording', buffer: audioData });
            }
        };
    }

    process(inputs) {
        if (!this.isRecording) {
            this.isRecording = true;
            this.recordings = [];
        }

        const input = inputs[0];
        const channelData = input[0];
        this.recordings.push(new Float32Array(channelData));

        return true;
    }

    getRecording() {
        const recordingLength = this.recordings.reduce((acc, curr) => acc + curr.length, 0);
        const audioData = new Float32Array(recordingLength);
        let offset = 0;

        for (const recording of this.recordings) {
            audioData.set(recording, offset);
            offset += recording.length;
        }

        // Convert to 16-bit PCM
        const pcmData = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        this.recordings = [];
        this.isRecording = false;
        return pcmData;
    }
}

registerProcessor('recorder-processor', RecorderProcessor);