import math
import pickle
import os

class BitStreamEncoder:
    def __init__(self, K: int, M: int, C: int, device="cuda", use_precomputed_combs=True, to_base_path='.'):
        self.K = K
        self.M = M
        self.C = C

        self.bits_for_rank = math.ceil(math.log2(math.comb(K, M)))
        self.bits_for_coeff = self.C # bits for one coeff
        
        self.device = device
        self.accumulated_bitstring = ''
        self.use_precomputed_combs = use_precomputed_combs
        self.combs_table = None

        if use_precomputed_combs:
            full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
            if not os.path.exists(full_path):
                BitStreamEncoder.precompute_comb_table(K, M, C, to_base_path)
            with open(full_path, "rb") as f:
                self.combs_table = pickle.load(f)['table']

    def _compute_rank(self, comb: list[int]) -> int:
        rank = 0
        prev = -1

        if self.use_precomputed_combs:
            for i in range(self.M):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                    rank += self.combs_table[x][i]
                prev = c_i

        else:
            for i in range(self.M):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                        rank += math.comb(self.K - x - 1, self.M - i - 1)
                prev = c_i
            
        return rank


    def add(self, comb: list[int], coeffs: list[int]):
        # sorting the comb to compute lex order and coeffs accordingly
        sorted_pairs = sorted(zip(comb, coeffs))
        sorted_comb, sorted_coeffs = zip(*sorted_pairs)
        sorted_comb = list(sorted_comb)
        sorted_coeffs = list(sorted_coeffs)

        rank = self._compute_rank(sorted_comb)
        
        iteration_bitstring = format(rank, f'0{self.bits_for_rank}b') + ''.join(format(coeff, f'0{self.bits_for_coeff}b') for coeff in sorted_coeffs)
        self.accumulated_bitstring += iteration_bitstring
        
        return iteration_bitstring

    def get_encoding(self) -> list[int]:
        return self.accumulated_bitstring

    def clear(self):
        self.accumulated_bitstring = ''
    
    def decode(self, encoding):
        expected_bits_per_iteration = self.bits_for_rank + self.bits_for_coeff * self.M
        assert expected_bits_per_iteration > 8 # otherwise there might be confusion with the padding to bytes
        
        len_without_padding = expected_bits_per_iteration * (len(encoding) // expected_bits_per_iteration)
        encoding = encoding[len(encoding) - len_without_padding:]
        encoding_iterations_split = [encoding[i:i+expected_bits_per_iteration] for i in range(0, len(encoding), expected_bits_per_iteration)]
        
        decoding = []
        for iteration_bits in encoding_iterations_split:
            rank = int(iteration_bits[:self.bits_for_rank], 2)
            coeffs = iteration_bits[self.bits_for_rank:]
            coeffs = [int(coeffs[i:i+self.bits_for_coeff], 2) for i in range(0, len(coeffs), self.bits_for_coeff)]

            # decode combination
            comb = []
            x = 0
            for i in range(self.M):
                while True:
                    value = self.combs_table[x][i] if self.use_precomputed_combs else math.comb(self.K - x - 1, self.M - i - 1)
                    if value > rank:
                        break
                    rank -= value
                    x += 1
                comb.append(x)
                x += 1
            
            decoding.append((comb, coeffs))
        
        return decoding

    def get_pkl_full_path(K, M, to_base=''):
        pkl_name = f"comb_K{K}_M{M}.pkl"
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "combs_files", pkl_name)

    @staticmethod
    def precompute_comb_table(K: int, M: int, C, to_base_path='.'):
        print(f"creating combinations file for M={M}. It is generated once and reused for all subsequent runs.")
        full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
        table = [[0 for i in range(M)] for x in range(K)]
    
        for x in range(K):
            for i in range(M):
                n = K - x - 1
                k = M - i - 1
                if n >= 0 and k >= 0 and n >= k:
                    table[x][i] = math.comb(n, k)
                else:
                    table[x][i] = 0
    
        with open(full_path, 'wb') as f:
            pickle.dump({
                'K': K,
                'M': M,
                'table': table
            }, f)
    