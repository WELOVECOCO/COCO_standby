from fastfft import FFTConvolver

conv = FFTConvolver()

result = conv.convolve([[1,2], [3,4]], [[0,1], [1,0]])

print(result)