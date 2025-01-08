from math import sin, pi
import struct

class WavGen:
    @staticmethod
    def generate_sine_samples(frequency=100, sample_rate=16000, duration=5):
        samples = []
        num_samples = int(duration * sample_rate)
        two_pi = 2 * pi
        max_amplitude = 32767.0
        
        # Pre-calculate coefficient to reduce computations in loop
        freq_coefficient = two_pi * frequency / sample_rate
        
        for i in range(num_samples):
            # Generate 16-bit sine wave sample (-32768 to 32767)
            # Using optimized calculation to reduce floating point operations
            sample = int(max_amplitude * sin(i * freq_coefficient))
            samples.append(sample)
            
        return samples



class AdpcmWavEncoder:
    # Index table for step size updates
    _index_table = [-1, -1, -1, -1, 2, 4, 6, 8,
                    -1, -1, -1, -1, 2, 4, 6, 8]

    # Step size lookup table
    _step_table = [
        7,     8,     9,    10,    11,    12,    13,    14,    16,    17,
       19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
       50,    55,    60,    66,    73,    80,    88,    97,   107,   118,
      130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
      337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
      876,   963,  1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
     2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
     5894,  6484,  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
    ]

    # Difference lookup table
    _diff_lookup = [
     1,  3,  5,  7,  9,  11,  13,  15,
    -1, -3, -5, -7, -9, -11, -13, -15
    ]

    def __init__(self):
        self.prev_sample = 0
        self.step_index = 0

    def encode_sample(self, sample):
        """Encode a single 16-bit PCM sample to 4-bit ADPCM."""
        # Calculate difference between current and previous sample
        delta = sample - self.prev_sample
        
        # Calculate the nibble value
        step = self._step_table[self.step_index]
        nibble = min(7, abs(delta) * 4 // step)
        if delta < 0:
            nibble += 8

        # Update predictor and step index
        diff = (step * self._diff_lookup[nibble]) // 8
        self.prev_sample += diff
        
        # Clamp predictor to 16-bit signed range
        if self.prev_sample > 32767:
            self.prev_sample = 32767
        elif self.prev_sample < -32768:
            self.prev_sample = -32768
            
        # Update step index
        self.step_index += self._index_table[nibble]
        if self.step_index < 0:
            self.step_index = 0
        elif self.step_index > 88:
            self.step_index = 88

        return nibble

    def encode_block(self, f, samples, block_align=1024):
        """Encode a block of samples and write to file.
        
        Args:
            f: Open file handle to write to
            samples: List of 16-bit PCM samples
            block_align: Block size in bytes (default 1024)
        """
        
        # Pad with zeros if less than 2041 samples
        if len(samples) < 2041:
            samples.extend([0] * (2041 - len(samples)))

        self.prev_sample = samples[0]

        # Write block header
        f.write(struct.pack('<h', self.prev_sample))  # Initial predictor
        f.write(struct.pack('<B', self.step_index))   # Initial index
        f.write(struct.pack('<B', 0))                 # Reserved byte
        
        self.prev_sample = samples[0]

        current_byte = 0
        byte_count = 0
        
        for sample in samples[1:]:
            encoded = self.encode_sample(sample)
            if byte_count % 2 == 0:
                current_byte = encoded
            else:
                current_byte |= (encoded << 4)
                f.write(struct.pack('B', current_byte))
            byte_count += 1
        
        if byte_count % 2 != 0:
            f.write(struct.pack('B', current_byte))




class WavWriter:
    @staticmethod
    def _write_riff_header(f):
        """Write RIFF WAV header and return position for size."""
        f.write(b'RIFF')
        size_pos = f.tell()
        f.write(struct.pack('<I', 0))  # Size placeholder
        f.write(b'WAVE')
        return size_pos
        
    @staticmethod
    def _write_fmt_pcm(f, sample_rate):
        """Write PCM format chunk."""
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Chunk size
        f.write(struct.pack('<H', 1))   # Format code (1 = PCM)
        f.write(struct.pack('<H', 1))   # Channels (1 = mono)
        f.write(struct.pack('<I', sample_rate))  # Sample rate
        bytes_per_sec = sample_rate * 2  # 16-bit = 2 bytes per sample
        f.write(struct.pack('<I', bytes_per_sec))  # Bytes per second
        f.write(struct.pack('<H', 2))   # Block align (2 bytes per sample)
        f.write(struct.pack('<H', 16))  # Bits per sample
        
    @staticmethod
    def _write_fmt_adpcm(f, sample_rate, block_align, samples_per_block):
        """Write ADPCM format chunk."""
        f.write(b'fmt ')
        f.write(struct.pack('<I', 20))  # Chunk size (20 for IMA ADPCM)
        f.write(struct.pack('<H', 17))  # Format code (17 = IMA ADPCM)
        f.write(struct.pack('<H', 1))   # Channels (1 = mono)
        f.write(struct.pack('<I', sample_rate))  # Sample rate
        f.write(struct.pack('<I', sample_rate))  # Average bytes per second
        f.write(struct.pack('<H', block_align))  # Block align
        f.write(struct.pack('<H', 4))   # Bits per sample
        f.write(struct.pack('<H', 2))   # Extra format bytes
        f.write(struct.pack('<H', samples_per_block))  # Samples per block
        
    @staticmethod
    def _write_data_header(f):
        """Write data chunk header and return positions."""
        f.write(b'data')
        data_size_pos = f.tell()
        f.write(struct.pack('<I', 0))  # Size placeholder
        data_start = f.tell()
        return data_size_pos, data_start
        
    @staticmethod
    def _update_sizes(f, size_pos, data_size_pos, data_start):
        """Update RIFF and data chunk sizes."""
        file_size = f.tell()
        f.seek(size_pos)
        f.write(struct.pack('<I', file_size - 8))  # RIFF chunk size
        f.seek(data_size_pos)
        f.write(struct.pack('<I', file_size - data_start))  # data chunk size
        
    @staticmethod
    def write_pcm(filename, samples, sample_rate=16000):
        """Write PCM samples to a WAV file.
        
        Args:
            filename (str): Output WAV filename
            samples (list): List of 16-bit PCM samples
            sample_rate (int): Sample rate in Hz
        """
        import struct
        
        with open(filename, 'wb') as f:
            size_pos = WavWriter._write_riff_header(f)
            WavWriter._write_fmt_pcm(f, sample_rate)
            data_size_pos, data_start = WavWriter._write_data_header(f)
            
            # Write sample data
            for sample in samples:
                f.write(struct.pack('<h', sample))
                
            WavWriter._update_sizes(f, size_pos, data_size_pos, data_start)
    
    @staticmethod        
    def write_adpcm(filename, samples, sample_rate=16000):
        """Write samples to a WAV file using IMA ADPCM encoding.
        
        Args:
            filename (str): Output WAV filename
            samples (list): List of 16-bit PCM samples to encode
            sample_rate (int): Sample rate in Hz
        """
        import struct
        
        block_align = 1024  # Block size in bytes
        samples_per_block = 2041  # Samples that fit in one block
        
        with open(filename, 'wb') as f:
            size_pos = WavWriter._write_riff_header(f)
            WavWriter._write_fmt_adpcm(f, sample_rate, block_align, samples_per_block)
            
            # Write fact chunk (required for compressed formats)
            f.write(b'fact')
            f.write(struct.pack('<I', 4))  # Chunk size
            f.write(struct.pack('<I', len(samples)))  # Number of samples
            
            data_size_pos, data_start = WavWriter._write_data_header(f)
            
            # Write blocks of encoded samples
            encoder = AdpcmWavEncoder()
            num_blocks = 0
            for i in range(0, len(samples), samples_per_block):
                block_samples = samples[i:i + samples_per_block]
                encoder.encode_block(f, block_samples, block_align)
                num_blocks += 1
                
            WavWriter._update_sizes(f, size_pos, data_size_pos, data_start)





sample_rate = 16000

# sin150 = WavGen.generate_sine_samples(150, sample_rate, 5)
# WavWriter.write_pcm('output_150_pcm.wav', sin150, sample_rate)
# WavWriter.write_adpcm('output_150_adpcm.wav', sin150, sample_rate)

sin500 = WavGen.generate_sine_samples(500, sample_rate, 5)
WavWriter.write_pcm('output_500_pcm.wav', sin500, sample_rate)
WavWriter.write_adpcm('output_500_adpcm.wav', sin500, sample_rate)