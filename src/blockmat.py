"""
Classes to deal with the manipulation and operations of large block diagonal
matrices and vectors in a way that:
    1. Is more efficient than using numpy as it doesn't store all the zeros 
       involved.
    2. Readily works for rectangular blocks (as long as the blocks are all the
       same shape).
"""
import numpy as np

class BlockMatrix:
    _trivial = False
    def __init__(self, mat, mode='block', nblock=None):
        """
        Representation of a block-diagonal matrix with off-diagonal zeros. The 
        blocks need not be square, but must have the same dimension as one
        another.

        Parameters
        ----------
        mat : 3D or 2D numpy array
            Pass either a block to repeat (2D), in which case must specify the 
            number of repeats n, or pass a list of the individual blocks to 
            repeat (3D), in which case the 2D blocks must each have the same
            shape.
        mode : str, optional
            If a 2D mat is passed, 'as-is' specifies the matrix as being
            complete, while 'block' specifies it as corresponding to a single
            block of the desired matrix. If a 3D matrix is passed, mode is
            ignored.
        nblock : int, optional
            Specifies the number of repeats if a single block is passed. If a 
            3D mat is passed, this argument is ignored.
        """
        # Check validity of inputs
        input_mat_shape = np.shape(mat)
        if not isinstance(mat, np.ndarray):
            raise ValueError("must pass mat as a numpy array.")

        if len(input_mat_shape) == 3:
            self.block_shape = np.shape(mat[0])
            for block in mat:
                if np.shape(block) != self.block_shape:
                    raise ValueError("shape of matrix blocks must be consistent.")
            nblock = input_mat_shape[0]

        elif len(input_mat_shape) == 2:
            if nblock is None:
                raise ValueError("must specify nblock when inputting either a single matrix block or the whole matrix.")
            if mode == 'block':
                self._trivial = True
                self.block_shape = input_mat_shape
                mat = np.array([mat for _ in range(nblock)])
            elif mode == 'as-is':
                if input_mat_shape[0]%nblock or input_mat_shape[1]%nblock:
                    raise ValueError("nblock must divide length of as-is vector.")
                self.block_shape = (int(input_mat_shape[0]/nblock), 
                                    int(input_mat_shape[1]/nblock))
                bs0, bs1 = self.block_shape
                mat = [mat[n*bs0:(n+1)*bs0,n*bs1:(n+1)*bs1] for n in range(nblock)]
                mat = np.array(mat)

        else:
            raise ValueError("mat must be either 2D or 3D.")
        
        self.mat_shape = tuple( np.array(self.block_shape) * nblock )
        self.nblock = nblock
        self._matrix = mat

    def __repr__(self) -> str:
        s = f"{self.block_shape} x {self.nblock} BlockMatrix"
        if self._trivial:
            s += " (trivial)"
        return s

    def __len__(self) -> int:
        return self.nblock * self.block_shape[0]

    def __matmul__(self, other):
        """
        Return a BlockMatrix representation of the product of self and other.

        Parameters
        ----------
        other : BlockMatrix or numpy.ndarray
            Must have the correct dimensionality for matrix multiplication. If 
            an ndarray is passed, will attempt to create a BlockMatrix with the 
            'block' flag or a BlockVector from it with the 'as-is' flag for 
            multiplication.
            These defaults correspond to my convenient use cases.
        """
        # If other is an ndarray, attempt to interpret it as a Block object.
        if isinstance(other, np.ndarray):
            if np.shape(other)[0] != self.mat_shape[1]:
                raise ValueError("incompatible matrix shapes.")
            if len(np.shape(other)) == 1:
                other = BlockVector(vec=other, mode='as-is', nblock=self.nblock)
            elif len(np.shape(other)) == 2:
                other = BlockMatrix(mat=other, mode='block', nblock=self.nblock)
            else:
                raise ValueError("ndarray of incompatible shape for matmul with this Block object.")

        # Check dimensionality is right.
        if self.mat_shape[1] != other.mat_shape[0]:
            raise ValueError("incompatible matrix shapes.")
        
        # Multiply matrices.
        if self._trivial and other._trivial:
            new_block = self.block[0] @ other.block[0]
            product = np.array([new_block for _ in range(self.nblock)])
        else:
            product = []
            for self_block, other_block in zip(self._matrix, other._matrix):
                product.append(self_block@other_block)
            product = np.array(product)
        
        if np.shape(product)[-1] == 1:
            return BlockVector(product)
        return BlockMatrix(product)
    
    def __add__(self, other):
        """
        Return a Blockmatrix representation of the sum of self and other.
        """
        # If other is an ndarray, attempt to interpret it as a Block object.
        if isinstance(other, np.ndarray):
            if np.shape(other)[0] != self.mat_shape[1]:
                raise ValueError("incompatible matrix shapes.")
            if len(np.shape(other)) == 1:
                other = BlockVector(vec=other, mode='as-is', nblock=self.nblock)
            elif len(np.shape(other)) == 2:
                other = BlockMatrix(mat=other, mode='block', nblock=self.nblock)
            else:
                raise ValueError("ndarray of incompatible shape for matmul with this Block object.")
        
        # Check dimensionality is right.
        if self.mat_shape != other.mat_shape:
            raise ValueError("incompatible matrix shapes.")
        
        sum = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            sum.append(self_block+other_block)
        sum = np.array(sum)
        if np.shape(sum)[-1] == 1:
            return BlockVector(sum)
        return BlockMatrix(sum)

    def __sub__(self, other):
        """
        Return a Blockmatrix representation of the difference of self and other.
        """
        # If other is an ndarray, attempt to interpret it as a Block object.
        if isinstance(other, np.ndarray) and np.shape(other)[0] == self.mat_shape[1]:
            if len(np.shape(other)) == 1:
                other = BlockVector(vec=other, mode='as-is', nblock=self.nblock)
            elif len(np.shape(other)) == 2:
                other = BlockMatrix(mat=other, mode='block', nblock=self.nblock)
            else:
                raise ValueError("ndarray of incompatible shape for matmul with this Block object.")
    
        # Check dimensionality is right.
        if self.mat_shape != other.mat_shape:
            raise ValueError("incompatible matrix shapes.")
        
        difference = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            difference.append(self_block-other_block)
        difference = np.array(difference)
        if np.shape(difference)[-1] == 1:
            return BlockVector(difference)
        return BlockMatrix(difference)
    
    def __mul__(self, other):
        """
        Define multiplication between Block matrices and floats/integers.
        """
        new_mat = self.block * other
        return BlockMatrix(new_mat, nblock=self.nblock)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Define division between Block matrices and floats/integers.
        """
        new_mat = self.block / other
        return BlockMatrix(new_mat, nblock=self.nblock)
    
    def __getitem__(self, key):
        """
        Define slicing for BlockMatrix objects. Returns a new BlockMatrix object 
        with each 2D array in the 3D array sliced according to the provided 
        slices.
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_slice = key
            new_matrix = [block[row_slice, col_slice] for block in self._matrix]
            return BlockMatrix(np.array(new_matrix), nblock=self.nblock)
        else:
            raise IndexError("Invalid index. Use a tuple of slices.")
    
    @property
    def matrix(self):
        """
        Return the full block-diagonal matrix.
        """
        full_mat = np.zeros(self.mat_shape)
        for n in range(self.nblock):
            b0 = self.block_shape[0]
            b1 = self.block_shape[1]
            full_mat[n*b0:(n+1)*b0, n*b1:(n+1)*b1] = self._matrix[n]
        return full_mat
    
    @property
    def block(self):
        """
        Return the list of matrix blocks. Can be used to easily access the nth
        block of the matrix.
        """
        return self._matrix
    
    @property
    def diag(self):
        """
        Return the diagonal of the matrix, as long as it's composed of square
        blocks.
        """
        if self.block_shape[0] != self.block_shape[1]:
            raise ValueError("diagonal of non-square matrix is undefined.")
        return np.diag(self.matrix)
    
    @property
    def T(self):
        """
        Return the transposed block-diagonal matrix.
        """
        new_mat = []
        for block in self._matrix:
            new_mat.append(block.T)
        new_mat = np.array(new_mat)
        new_blockmat = BlockMatrix(mat=new_mat)
        return new_blockmat
    
    @property
    def inv(self):
        """
        Return the inverted block-diagonal matrix if the matrix is square.
        """
        if self.block_shape[0] != self.block_shape[1]:
            raise ValueError("inverse of non-square matrix is undefined.")
        if self._trivial:
            new_block = np.linalg.inv(self._matrix[0])
            new_mat = [new_block for _ in range(self.nblock)]
        else:
            new_mat = []
            for block in self._matrix:
                new_mat.append(np.linalg.inv(block))
        new_mat = np.array(new_mat)
        new_blockmat = BlockMatrix(mat=new_mat)
        return new_blockmat


class BlockVector(BlockMatrix):
    def __init__(self, vec, mode='as-is', nblock=None):
        """
        Representation of a block vector. Stores it as a series of [N,1] block
        matrices under-the-hood.

        Parameters
        ----------
        vec : 1D, 2D or 3D numpy array
            Some representation of the block diagonal vector to construct.
            Either the vector itself (1D), a single block of the vector (1D),
            all blocks of the vector in a list (2D), or the full under-the hood
            representation as a list of column vectors (3D).
        mode : str, optional
            If a 1D vec is passed, 'as-is' specifies the vector as being
            complete, while 'block' specifies it as corresponding to a single
            block of the desired vector. If a 2D vector is passed, mode is
            ignored.
        nblock : int, optional
            If a 1D vec is passed and mode is 'as-is', specifies the number of 
            blocks that the vector is composed of. If mode is 'block', specifies
            the number of desired blocks of the full vector. If a 2D vector is 
            passed, nblock is ignored.
        """
        # Check inputs and separate vector blocks.
        if len(np.shape(vec)) == 1:
            if mode == 'as-is':
                if nblock is None:
                    raise ValueError("must specify the number of blocks in the complete vector.")
                if len(vec) % nblock:
                    raise ValueError("nblock must divide length of as-is vector.")
                self.block_len = int(len(vec) / nblock)
                split_vec = [vec[n*self.block_len:(n+1)*self.block_len] for n in range(nblock)]
                mat = np.array([np.array([vector]).T for vector in split_vec])
            
            elif mode == 'block':
                if nblock is None:
                    raise ValueError("must specify the number of times to repeat vector block.")
                self.block_len = len(vec)
                one_block = np.array([vec]).T
                mat = np.array([one_block]*nblock)

            else:
                raise ValueError("invalid mode specified.")
        
        elif len(np.shape(vec)) == 2:
            nblock = len(vec)
            self.block_len = len(vec[0])
            for v in vec:
                if len(v) != self.block_len:
                    raise ValueError("length of vector blocks must be consistent.")
            mat = np.array([np.array([v]).T for v in vec])
        
        elif len(np.shape(vec)) == 3:
            if np.shape(vec)[-1] != 1:
                raise ValueError("expecting a list of column vectors. Are you inputting a list of matrices?")
            nblock = len(vec)
            self.block_len = len(vec[0])
            for v in vec:
                if len(v) != self.block_len:
                    raise ValueError("shape of vector blocks must be consistent.")
            mat = np.array(vec)
        
        else:
            raise ValueError("invalid vec shape.")

        self.nblock = nblock
        super().__init__(mat, nblock)

    def __repr__(self):
        return f"({(self.block_len)},) x {self.nblock} BlockVector"

    def __add__(self, other):
        """
        Return a BlockVector representation of the sum of self and other.
        """
        # Check dimensionality is right.
        if not isinstance(other, BlockVector):
            raise TypeError(f"unsupported operand type for +: 'BlockVector' and '{type(other)}'.")
        if self.mat_shape != other.mat_shape:
            raise ValueError("incompatible vector shapes.")
        
        sum = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            sum.append(self_block+other_block)
        return BlockVector(sum)
    
    def __truediv__(self, other):
        """
        Define division between Block vectors and floats/integers.
        """
        new_vec = self.block / other
        return BlockVector(new_vec)
    
    def __pow__(self, other):
        """
        Define raising to the power between Block vectors and floats/integers.
        """
        new_vec = self.block ** other
        return BlockVector(new_vec)
    
    def __getitem__(self, key):
        """
        Define slicing for BlockVector objects. Returns a new BlockVector object 
        with each block in the vector sliced according to the provided slice.
        """
        if isinstance(key, int):
            return self[key:key+1].vector
        if isinstance(key, slice):
            new_matrix = [block[key, :] for block in self._matrix]
            return BlockVector(new_matrix, nblock=self.nblock)
        else:
            raise IndexError("Invalid index. Use a slice.")
    
    @property
    def vector(self):
        """
        Return the full block vector.
        """
        return self._matrix.flatten()
    
    @property
    def block(self):
        """
        Return the list of vector blocks. Can be used to easily access the nth
        block of the vector.
        """
        return np.array(np.split(self.vector, self.nblock))

    @property
    def vec_len(self):
        """
        Return the total length of the block vector.
        """
        return len(self.vector)
    
    @property
    def vec_block_len(self):
        """
        Return the length of the vector's block.
        """
        return len(self.block[0])
