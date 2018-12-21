kernel void histogram(
	global unsigned char *image,
	unsigned int _size,
	global unsigned int *result)
{
	int color = get_global_id(0);
	global unsigned int *ptr = result + color * 256;
	for (unsigned int i = color; i < _size; i+= 3) {
		unsigned int index = image[i];
		ptr[index]++;
	}
}
