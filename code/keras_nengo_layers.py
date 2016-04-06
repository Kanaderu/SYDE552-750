import numpy as np
from nengo.processes import Process
from nengo.params import EnumParam, IntParam, NdarrayParam, TupleParam

#Code adapted from nengo_deeplearning by Eric Hunsberger
#https://github.com/hunse/cuda-convnet2/blob/softlif/run_nengo.py

class Conv2d(Process):

    shape_in = TupleParam('shape_in', length=3)
    shape_out = TupleParam('shape_out', length=3)
    stride = TupleParam('stride', length=2)
    padding = TupleParam('padding', length=2)
    filters = NdarrayParam('filters', shape=('...',))
    biases = NdarrayParam('biases', shape=('...',), optional=True)

    def __init__(self, shape_in, filters, biases=None, stride=1, padding=0, activation='linear'):  # noqa: C901
        from nengo.utils.compat import is_iterable, is_integer

        self.shape_in = tuple(shape_in)
        self.filters = filters
        self.stride = stride if is_iterable(stride) else [stride] * 2
        self.padding = padding if is_iterable(padding) else [padding] * 2
        self.activation=activation

        nf = self.filters.shape[0]
        nxi, nxj = self.shape_in[1:]
        si, sj = self.filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.stride
        nyi = 1 + max(int(np.ceil((2*pi + nxi - si) / float(sti))), 0)
        nyj = 1 + max(int(np.ceil((2*pj + nxj - sj) / float(stj))), 0)
        self.shape_out = (nf, nyi, nyj)

        self.biases = biases if biases is not None else None
        if self.biases is not None:
            if self.biases.size == 1:
                self.biases.shape = (1, 1, 1)
            elif self.biases.size == np.prod(self.shape_out):
                self.biases.shape = self.shape_out
            elif self.biases.size == self.shape_out[0]:
                self.biases.shape = (self.shape_out[0], 1, 1)
            elif self.biases.size == np.prod(self.shape_out[1:]):
                self.biases.shape = (1,) + self.shape_out[1:]

        super(Conv2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        filters = self.filters
        local_filters = filters.ndim == 6
        biases = self.biases
        shape_in = self.shape_in
        shape_out = self.shape_out

        nxi, nxj = shape_in[-2:]
        nyi, nyj = shape_out[-2:]
        nf = filters.shape[0]
        si, sj = filters.shape[-2:]
        pi, pj = self.padding
        sti, stj = self.stride

        def step_conv2d(t, x):
            x = x.reshape(shape_in)
            y = np.zeros(shape_out)

            for i in range(nyi):
                for j in range(nyj):
                    i0 = i*sti - pi
                    j0 = j*stj - pj
                    i1, j1 = i0 + si, j0 + sj
                    sli = slice(max(-i0, 0), min(nxi + si - i1, si))
                    slj = slice(max(-j0, 0), min(nxj + sj - j1, sj))
                    w = (filters[:, i, j, :, sli, slj] if local_filters else
                         filters[:, :, sli, slj])
                    xij = x[:, max(i0, 0):min(i1, nxi),
                            max(j0, 0):min(j1, nxj)]
                    y[:, i, j] = np.dot(w.reshape(nf, -1), xij.ravel())

            if biases is not None:
                    y += biases

            if self.activation =='linear':
                return y.ravel()
            elif self.activation == 'relu': #element-wise
                rect=np.maximum(y.ravel(),np.zeros((y.ravel().shape[0])))
                return rect

        return step_conv2d

#The pooling bug is somewhere in this class, most likely in the slicing syntax
class Pool2d(Process):
    shape_in = TupleParam('shape_in', length=3)
    shape_out = TupleParam('shape_out', length=3)
    size = IntParam('size', low=1)
    stride = IntParam('stride', low=1)
    kind = EnumParam('kind', values=('avg', 'max'))

    def __init__(self, shape_in, size, stride=None, kind='avg'):
        self.shape_in = shape_in
        self.size = size
        self.stride = stride if stride is not None else size
        self.kind = kind

        nc, nxi, nxj = self.shape_in
        # nyi = 1 + int(np.ceil(float(nxi - size) / self.stride))
        nyi = int(np.floor((1+max(int((nxi-1)/float(stride)),0))/size))
        # nyj = 1 + int(np.ceil(float(nxj - size) / self.stride))
        nyj = int(np.floor((1+max(int((nxj-1)/float(stride)),0))/size))
        self.shape_out = (nc, nyi, nyj)
        # print self.shape_in, self.shape_out
        super(Pool2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)
        nc, nxi, nxj = self.shape_in
        nc, nyi, nyj = self.shape_out
        s = self.size
        st = self.stride
        kind = self.kind
        #(nyi - 1) * st + s, (nyj - 1) * st + s
        nxi2, nxj2 = nyi, nyj

        def step_pool2d(t, x):
            x = x.reshape(nc, nxi, nxj)
            y = np.zeros((nc, nyi, nyj), dtype=x.dtype)
            n = np.zeros((nyi, nyj))

            for i in range(s):
                for j in range(s):
                    xij = x[:, i:nxi2:st, j:nxj2:st]
                    ni, nj = xij.shape[-2:]
                    if kind == 'max':
                        y[:, :ni, :nj] = np.maximum(y[:, :ni, :nj], xij)
                    elif kind == 'avg':
                        y[:, :ni, :nj] += xij
                        n[:ni, :nj] += 1
                    else:
                        raise NotImplementedError(kind)

            if kind == 'avg':
                y /= n

            return y.ravel()

        return step_pool2d


class FeatureMap2d(Process):

    shape_in = TupleParam('shape_in', length=3)
    shape_out = TupleParam('shape_out', length=3)

    def __init__(self, shape_in, activation='linear', recurrent='none',rad=1):
        from nengo.utils.compat import is_iterable, is_integer

        self.shape_in = tuple(shape_in)
        self.shape_out = tuple(shape_in)
        self.activation=activation
        self.recurrent=recurrent
        self.rad=rad

        super(FeatureMap2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        def step_conv2d(t, x):
            x = x.reshape(self.shape_in)
            
            if self.recurrent == 'none':
            	y = x
            #center-surround inhibition: in each feature map, add the activity of x[i][j] to itself,
            #then take nearest neighbors within radius pixelsand subtract their activity from x[i][j]
            #ignore borders
            elif self.recurrent == 'center-surround':
                dx=np.zeros(self.shape_in)
                for fm in np.arange(0,x.shape[0]):
                    for i in np.arange(self.rad,x.shape[1]-self.rad):
                        for j in np.arange(self.rad,x.shape[2]-self.rad):
                            dx[fm][i][j]=-2.5*self.rad*np.average(x[fm][i-self.rad:i+self.rad+1]\
                                [:,j-self.rad:j+self.rad+1])+\
                                2*x[fm][i][j]
                y = x+dx
            if self.activation =='linear':
                return y.ravel()

        return step_conv2d

class Dense_1d(Process):

    def __init__(self, shape_in, shape_out, weights, biases, activation='linear'): 
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.weights = weights
        self.biases = biases
        self.activation = activation
        super(Dense_1d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        def step_dense1d(t, x):
            x = x.reshape(self.shape_in)
            x_post = np.dot(x,self.weights)
            
            if self.activation == 'softmax':
                y = np.array([np.exp(xi)/np.sum((np.exp(x_post))) for xi in x_post])
            elif self.activation == 'relu':
                y = np.maximum(x_post,np.zeros((x_post.shape)))
            elif self.activation == 'linear':
                y = x_post

            if self.biases is not None:         
                y += self.biases
            return y.ravel()

        return step_dense1d


class Flatten(Process):

	def __init__(self, shape_in, shape_out):
		self.shape_in = shape_in
		self.shape_out = shape_out
		super(Flatten, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)

		def step_flatten(t, x):
			x = x.reshape(self.shape_in)
			y = x.ravel()
			return y

		return step_flatten


class Dropout(Process): #does nothing

    def __init__(self, shape_in, shape_out):
        self.shape_in = shape_in
        self.shape_out = shape_out
        super(Dropout, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        def step_flatten(t, x):
            x=x.reshape(self.shape_in)
            y=x.ravel()
            return y

        return step_flatten


class Sal_F(Process):

	def __init__(self, shape_in, shape_out):  # sprq
		self.shape_in = tuple(shape_in)
		self.shape_out = shape_out
		super(Sal_F, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)
		shape_in = self.shape_in
		shape_out = self.shape_out

		def step_F(t, x):
			x = x.reshape(shape_in)
			# y = np.sum(x,axis=(1,2))
			y = np.average(x,axis=(1,2))
			return y.ravel()

		return step_F


class Sal_C(Process):

	def __init__(self, shape_in, shape_out, competition='softmax'):	#5643
		self.shape_in = shape_in
		self.shape_out = shape_out
		self.competition = competition
		super(Sal_C, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)

		def step_C(t, x):
			if self.competition == 'softmax':
				y = np.divide(np.exp(x),np.sum((np.exp(x))))
			elif self.competition == 'none':
				y=x
			#TODO - Walther and Koch inhibitory competition
			#TODO - BG-type inhibitory competition
			return y.ravel()

		return step_C


class Sal_B_near(Process):

	def __init__(self, shape_in, shape_out, feedback_near, k_FB_near): #yeahright
		self.shape_in = shape_in
		self.shape_out = tuple(shape_out)
		self.feedback_near = feedback_near
		self.k_FB = k_FB_near #keeps feedback from exploding
		super(Sal_B_near, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)

		def step_B_near(t, x):
			if self.feedback_near == 'constant':
				#(n_FM,x,y) shape array filled with C value for corresponding feature map
				xy_shape=self.shape_out[1:]
				y=np.array([np.ones((xy_shape)) * xi for xi in x])
			elif self.feedback_near == 'none':
				y=np.zeros((self.shape_out))
			#TODO - feedback proportional to activation of each x,y unit in each FM
			#TODO - feedback with modulatory inputs
			y*=self.k_FB
			return y.ravel()

		return step_B_near


class Sal_B_far(Process):

	def __init__(self, shape_in, shape_out, weight, feedback_far, k_FB_far): #yeahright
		self.shape_in = shape_in
		self.shape_out = shape_out
		self.W = weight
		self.feedback_far = feedback_far
		self.k_FB=k_FB_far
		super(Sal_B_far, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)

		def step_B_far(t, x):
			x = x.reshape(self.shape_in)
			if self.feedback_far == 'dense_inverse':
				y=np.dot(x,np.linalg.pinv(self.W)) #the matrix inversion between salience maps
			elif self.feedback_far == 'none':
				y=np.zeros((self.shape_out))
			y*=self.k_FB
			return y.ravel()

		return step_B_far
