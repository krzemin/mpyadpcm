from math import sin, pi
from array import array
import struct

class WavGen:
    @staticmethod
    def generate_sine_samples(frequency=100, sample_rate=16000, duration=5):
        num_samples = int(duration * sample_rate)
        two_pi = 2 * pi
        max_amplitude = 32767.0
        
        # Create pre-allocated array of 16-bit signed integers ('h' type code)
        samples = array('h', [0] * num_samples)
        
        # Pre-calculate coefficient to reduce computations in loop
        freq_coefficient = two_pi * frequency / sample_rate
        
        for i in range(num_samples):
            # Generate 16-bit sine wave sample (-32768 to 32767)
            # Using optimized calculation to reduce floating point operations
            samples[i] = int(max_amplitude * sin(i * freq_coefficient))
            
        return samples

    @staticmethod
    def generate_rect_samples(frequency=100, sample_rate=16000, duration=5, amplitude=1.0):
        num_samples = int(duration * sample_rate)
        max_amplitude = 32767.0
        scaled_amplitude = int(max_amplitude * amplitude)
        
        # Create pre-allocated array of 16-bit signed integers ('h' type code)
        samples = array('h', [0] * num_samples)
        
        # Pre-calculate period in samples
        samples_per_period = sample_rate / frequency
        half_period = samples_per_period / 2
        
        for i in range(num_samples):
            # Generate rectangular wave by checking position in period
            if (i % samples_per_period) < half_period:
                samples[i] = scaled_amplitude
            else:
                samples[i] = -scaled_amplitude
            
        return samples


# Index table for step size updates
_adpcm_index_table = [-1, -1, -1, -1, 2, 4, 6, 8,
                    -1, -1, -1, -1, 2, 4, 6, 8]
# Step size lookup table
_adpcm_step_table = [
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
_adpcm_diff_lookup = [
     1,  3,  5,  7,  9,  11,  13,  15,
    -1, -3, -5, -7, -9, -11, -13, -15
    ]


class AdpcmWavEncoder:

    def __init__(self):
        self.prev_sample = 0
        self.step_index = 0

    def encode_sample(self, sample):
        """Encode a single 16-bit PCM sample to 4-bit ADPCM."""
        # Calculate difference between current and previous sample
        delta = sample - self.prev_sample
        
        # Calculate the nibble value
        step = _adpcm_step_table[self.step_index]
        nibble = min(7, abs(delta) * 4 // step)
        if delta < 0:
            nibble += 8

        # Update predictor and step index
        self.prev_sample += (step * _adpcm_diff_lookup[nibble]) // 8
        
        # Clamp predictor to 16-bit signed range
        if self.prev_sample > 32767:
            self.prev_sample = 32767
        elif self.prev_sample < -32768:
            self.prev_sample = -32768
            
        # Update step index
        self.step_index += _adpcm_index_table[nibble]
        if self.step_index < 0:
            self.step_index = 0
        elif self.step_index > 88:
            self.step_index = 88

        return nibble

    def encode_block(self, samples, block_align=1024):
        """Encode a block of samples and write to file.
        
        Args:
            samples: List of 16-bit PCM samples
            block_align: Block size in bytes (default 1024)
        """
        
        self.prev_sample = samples[0]

        # Create temporary byte array for the block
        block_bytes = bytearray(block_align)
        struct.pack_into('<h', block_bytes, 0, self.prev_sample)
        block_bytes[2] = self.step_index
        block_bytes[3] = 0
        
        byte_pos = 4
        i = 0

        while byte_pos < block_align:
            i += 1
            nibble1 = self.encode_sample(samples[i] if i < len(samples) else 0)
            i += 1
            nibble2 = self.encode_sample(samples[i] if i < len(samples) else 0)
            block_bytes[byte_pos] = nibble1 | (nibble2 << 4)
            byte_pos += 1
       
        return block_bytes

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
            f.write(samples)
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
                block_bytes = encoder.encode_block(block_samples, block_align)
                f.write(block_bytes)
                num_blocks += 1
                
            WavWriter._update_sizes(f, size_pos, data_size_pos, data_start)



sample_rate = 16000

sin150 = WavGen.generate_sine_samples(150, sample_rate, 5)
WavWriter.write_pcm('output_150_pcm.wav', sin150, sample_rate)
WavWriter.write_adpcm('output_150_adpcm.wav', sin150, sample_rate)

sin500 = WavGen.generate_sine_samples(500, sample_rate, 5)
WavWriter.write_pcm('output_500_pcm.wav', sin500, sample_rate)
WavWriter.write_adpcm('output_500_adpcm.wav', sin500, sample_rate)

rect200 = WavGen.generate_rect_samples(200, sample_rate, 5)
WavWriter.write_pcm('output_rect200_pcm.wav', rect200, sample_rate)
WavWriter.write_adpcm('output_rect200_adpcm.wav', rect200, sample_rate)
