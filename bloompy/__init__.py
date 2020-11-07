import math
import copy
import mmh3
import hashlib
from cmath import log

#need to install bitarray: pip install bitarray
from bitarray import bitarray
from struct import unpack, pack


name = 'bloompy'
__all__ = ['BloomFilter','CountingBloomFilter','ScalableBloomFilter','SCBloomFilter','get_filter_fromfile']
__version__='0.1.1'

IN_SEP = b'&&&&'
OUT_SEP = b'####'
FMT = '4sfiii'
_FMT ='fiii'

KIND = {
		1:'BloomFilter',
		2:'CountingBloomFilter',
		3:'ScalableBloomFilter',
		4:'SCBloomFilter'
	}


def primes(max_num):
	'''
	get prime numbers up to max_num
	'''
	if max_num<=1:
		raise ValueError('max_num should bigger than 1.')
	_ =[]
	for i in range(2,max_num):
	    flag = True
	    for j in range(2, i):
	        if i % j == 0:
	            flag = False
	            break
	    if flag:
	        _.append(i)
	return _


def get_filter_fromfile(path,check=None):
	with open(path, 'rb') as f:
		_bytes = f.read()
	if OUT_SEP in _bytes:
		_all_bytes = [i for i in _bytes.split(OUT_SEP) if i]
		_f_bytes = _all_bytes[:-1]
		_k_bytes = _all_bytes[-1]
		*args,kind = unpack(_FMT,_k_bytes)
		if check and check!=kind:
			raise TypeError(f'Not a {KIND[check]} to get.'
							f'It\'s a {KIND[kind]}')
		filters = [_get_filter_frombytes(i)
				   for i in _f_bytes]
		filter = eval(KIND[kind])(*args)
		filter.filters=filters
		return filter
	else:
		return _get_filter_frombytes(_bytes,check)

def _get_filter_frombytes(_bytes,check=None):
	_bits = _bytes.split(IN_SEP)[0]
	_args = _bytes.split(IN_SEP)[-1]
	*args, count, kind = unpack(_FMT, _args)
	if check and check != kind:
		raise TypeError(f'Not a {KIND[check]} to get.'
						f'It\'s a {KIND[kind]}')
	_bitarray = bitarray()
	_bitarray.frombytes(_bits)
	filter = eval(KIND[kind])(*args)
	filter.bit_array = _bitarray
	filter.count = count
	return filter


class BloomFilter(object):

	def __init__(self,error_rate,element_num=None,bit_num=None):
		self.count = 0
		self._c = element_num
		self.error_rate = error_rate
		self._b = bit_num
		self._install(error_rate,element_num,bit_num)

	def _install(self,error_rate,element_num,bit_num):
		if not error_rate or error_rate < 0:
			raise ValueError('error rate should be a positive value.')
		if bit_num and element_num:
			raise ValueError('Multi arguments are given,'
							 'needs 2 args,but got 3.')
		elif bit_num:
			element_num = -1 * bit_num * (log(2.0)*log(2.0)) \
						  / log(error_rate)
		elif element_num:
			bit_num = -1 * element_num * log(error_rate) \
					  / (log(2.0)*log(2.0))
		else:
			raise ValueError('Function arguments missing.'
							 '2 arguments should be given at least.')

		self.bit_num = self.__align_4bytes(bit_num.real)
		self.element_num = int(math.ceil(element_num.real))
		self.capacity = self.element_num if not self._c else self._c
		self.hash_num = int(math.ceil((log(2.0) * self.bit_num
									   / element_num).real))
		self.bit_array = bitarray(self.bit_num)
		self.bit_array.setall(False)
		self.seeds = primes(200)

	def add(self,element):
		if self._at_half_fill():
			raise IndexError('BloomFilter is at capacity.')
		i = 0
		element = self._to_str(element)
		for _ in range(self.hash_num):
			hashed_value = mmh3.hash(element,
									 self.seeds[_])%self.bit_num
			i += self.bit_array[hashed_value]
			self.bit_array[hashed_value]=1
		if i == self.hash_num:
			return True
		else:
			self.count += 1
			return False

	def copy(self):
		new_filter = BloomFilter(self.error_rate,
								 self.capacity, self._b)
		new_filter.bit_array = self.bit_array.copy()
		return new_filter

	def exists(self,element):
		element = self._to_str(element)
		for _ in range(self.hash_num):
			hashed_value = mmh3.hash(element,
									 self.seeds[_])%self.bit_num
			if self.bit_array[hashed_value] == 0:
				return False
		return True

	def to_pack(self):
		b_bitarray = self.bit_array.tobytes()
		b_args = pack(FMT,IN_SEP,self.error_rate,
					  self.capacity,self.count,1)
		return b_bitarray+b_args

	def tofile(self,path,mode='wb'):
		with open(path,mode) as f:
			f.write(self.to_pack())

	@classmethod
	def fromfile(cls,path):
		return get_filter_fromfile(path,1)

	def _option(self, other,opt):
		_ = {
			'or'	:	'__or__',
			'and' 	:	'__and__'
		}
		if self.capacity != other.capacity or \
				self.error_rate != other.error_rate or \
				self.bit_num != other.bit_num:
			raise ValueError(
			    "Both filters must have equal initial arguments:"
				"element_num、bit_num、error_rate")
		new_bloom = self.copy()
		new_bloom.bit_array = getattr(new_bloom.bit_array,
									  _[opt])(other.bit_array)
		return new_bloom

	def _to_str(self, element):
		_e_class = element.__class__.__name__
		_str = str(element)+_e_class
		bytes_like = bytes(_str, encoding='utf-8') if \
			isinstance(_str, str) else _str
		b_md5 = hashlib.md5(bytes_like).hexdigest()
		return b_md5

	def _at_half_fill(self):
		return self.bit_array.count()/self.bit_array.length()*1.0>=0.5

	def __align_4bytes(self,bit_num):
		return int(math.ceil(bit_num/32))*32

	def __or__(self, other):
		return self._option(other,'or')

	def __and__(self, other):
		return self._option(other,'and')

	def __contains__(self, item):
		return self.exists(item)

	def __len__(self):
		return self.count


class CountingBloomFilter(BloomFilter):

	def __init__(self,error_rate=0.001,element_num=10**4,bit_num=None):
		super(CountingBloomFilter,self).__init__(error_rate,element_num,bit_num)
		self._bit_array = self.bit_array
		self.bit_array = self._bit_array*4

	def add(self,element):
		'''
		query the element status in the filter and add it into the filter
		'''
		if self.count >= self.capacity:
			raise IndexError('BloomFilter is at capacity.')
		_element = self._to_str(element)
		i = 0
		for _ in range(self.hash_num):
			hashed_value = mmh3.hash(_element,
									 self.seeds[_]) % self.bit_num
			raw_value = self._get_bit_value(hashed_value)
			if raw_value > 0:
				i+=1
			self._set_bit_value(hashed_value,raw_value+1)
		if i == self.hash_num:
			return True
		else:
			self.count += 1
			return False

	def delete(self,element):
		'''
		query the element status in the filter and delete it from the filter
		'''
		if self.exists(element):
			_element = self._to_str(element)
			for _ in range(self.hash_num):
				hashed_value = mmh3.hash(_element,
										 self.seeds[_]) % self.bit_num
				raw_value = self._get_bit_value(hashed_value)
				self._set_bit_value(hashed_value, raw_value - 1)
			self.count-=1
			return True
		return False

	def exists(self,element):
		_element = self._to_str(element)
		for _ in range(self.hash_num):
			hashed_value = mmh3.hash(_element,
									 self.seeds[_])%self.bit_num
			if self._get_bit_value(hashed_value) <= 0:
				return False
		return True

	def copy(self):
		new_filter = CountingBloomFilter(self.error_rate,
										 self.capacity, self._b)
		new_filter.bit_array = self.bit_array.copy()
		return new_filter

	def to_pack(self):
		b_bitarray = self.bit_array.tobytes()
		b_args = pack(FMT,IN_SEP,self.error_rate,
					  self.capacity,self.count,2)
		return b_bitarray+b_args

	@classmethod
	def fromfile(cls,path):
		return get_filter_fromfile(path, 2)

	def _overflow(self):
		'''if we set 4 bits to represent a abstract bit
		 of the standard BloomFilter.The max overflow probability
		 is (asume that binary 1111 (int:16) equals j16)
		 p(j16)<=bit_num*(e*ln2/16)**16 .
		 the value is pretty small so we can use 4 bits to
		  represent one bit for common usage.'''
		max_overflow_probability = (math.exp(1)*math.log(2)/16)\
								   **16*self.bit_num
		return max_overflow_probability

	def _set_bit_value(self,index:int,value:int):
		_bin  = [int(i) for i in self._to_bin(value)]
		start = index*4
		end = 4*(index+1)
		for i in  range(start,end):
			self.bit_array[i] = _bin[i%4]

	def _get_bit_value(self,index:int)->int:
		start = index*4
		end = (index+1)*4
		_bits = self.bit_array[start:end]
		_v = int(_bits.to01(),2)
		return _v

	def _to_bin(self,value:int)->str:
		_bin = bin(value)[2:]
		if len(_bin)<4:
			_bin = '0'*(4-len(_bin))+_bin
		return _bin


class ScalableBloomFilter(object):
	'''
	forked from pybloom(https://github.com/jaybaird/python-bloomfilter).
	'''
	SMALL_SET_GROWTH = 2  # slower, but takes up less memory
	LARGE_SET_GROWTH = 4  # faster, but takes up more memory faster

	def __init__(self,  error_rate=0.001,initial_capacity=10**4,
                 mode=SMALL_SET_GROWTH):
		"""Implements a space-efficient probabilistic data structure that
		grows as more items are added while maintaining a steady false
		positive rate
		initial_capacity
			the initial capacity of the filter
		error_rate
			the error_rate of the filter returning false positives. This
			determines the filters capacity. Going over capacity greatly
			increases the chance of false positives.
		mode
			can be either ScalableBloomFilter.SMALL_SET_GROWTH or
			ScalableBloomFilter.LARGE_SET_GROWTH. SMALL_SET_GROWTH is slower
			but uses less memory. LARGE_SET_GROWTH is faster but consumes
			memory faster.
		"""
		self.filters = []
		self.scale = mode
		self.ratio = 0.9
		self.initial_capacity = initial_capacity
		self.error_rate = error_rate

	def add(self, key):
		"""Adds a key to this bloom filter.
		If the key already exists in this filter it will return True.
		Otherwise False.
		"""
		if key in self:
			return True
		if not self.filters:
			filter = BloomFilter(self.error_rate * self.ratio,
								 element_num=self.initial_capacity,)
			self.filters.append(filter)
			filter.add(key)
		else:
			filter = self.filters[-1]
			try:
				filter.add(key)
			except IndexError:
				filter = BloomFilter(
					element_num=filter.capacity * self.scale,
					error_rate=filter.error_rate * self.ratio)
				self.filters.append(filter)
				filter.add(key)
		return False

	def union(self, other):
		""" Calculates the union of the underlying classic bloom filters and returns
		a new scalable bloom filter object."""
		if self.scale != other.scale or \
				self.initial_capacity != other.initial_capacity or \
				self.error_rate != other.error_rate :
			raise ValueError("Unioning two scalable bloom filters requires \
			both filters to have both the same mode, initial capacity and error rate")
		if len(self.filters) > len(other.filters):
			larger_sbf = copy.deepcopy(self)
			smaller_sbf = other
		else:
			larger_sbf = copy.deepcopy(other)
			smaller_sbf = self
		# Union the underlying classic bloom filters
		new_filters = []
		for i in range(len(smaller_sbf.filters)):
			new_filter = larger_sbf.filters[i] | smaller_sbf.filters[i]
			new_filters.append(new_filter)
		for i in range(len(smaller_sbf.filters), len(larger_sbf.filters)):
			new_filters.append(larger_sbf.filters[i])
		larger_sbf.filters = new_filters
		return larger_sbf

	@property
	def capacity(self):
		"""Returns the total capacity for all filters in this SBF"""
		return sum(f.capacity for f in self.filters)

	@property
	def count(self):
		"""Returns the total number of elements stored in this SBF"""
		return sum(f.count for f in self.filters)

	def to_pack(self):
		b_pack = [i.to_pack() for i in self.filters]
		_k = pack(_FMT,self.error_rate,self.initial_capacity,self.scale,3)
		b_pack.append(_k)
		return OUT_SEP.join(b_pack)+OUT_SEP

	def tofile(self,path,mode='wb'):
		with open(path,mode) as f:
			f.write(self.to_pack())

	@classmethod
	def fromfile(cls,path):
		return get_filter_fromfile(path,3)

	def __or__(self, other):
		return self.union(other)

	def __contains__(self, key):
		"""Tests a key's membership in this bloom filter.
		"""
		for f in reversed(self.filters):
			if key in f:
				return True
		return False

	def __len__(self):
		return len(self.filters)


class SCBloomFilter(ScalableBloomFilter):

	def add(self, key):
		"""Adds a key to this bloom filter.
		If the key already exists in this filter it will return True.
		Otherwise False.
		"""
		if key in self:
			return True
		if not self.filters:
			filter = CountingBloomFilter(self.error_rate * self.ratio,
								 element_num=self.initial_capacity,)
			self.filters.append(filter)
			filter.add(key)
		else:
			for _,filter in enumerate(self.filters):
				try:
					filter.add(key)
					break
				except IndexError:
					if _+1 == len(self.filters):
						filter = CountingBloomFilter(
							element_num=filter.capacity * self.scale,
							error_rate=filter.error_rate * self.ratio)
						self.filters.append(filter)
						filter.add(key)
					else:
						continue
		return False

	def delete(self,key):
		for i in self.filters:
			if i.delete(key):
				return True
		return False

	def to_pack(self):
		b_pack = [i.to_pack() for i in self.filters]
		_k = pack(_FMT,self.error_rate,self.initial_capacity,self.scale,4)
		b_pack.append(_k)
		return OUT_SEP.join(b_pack)+OUT_SEP

	@classmethod
	def fromfile(cls,path):
		return get_filter_fromfile(path,4)
