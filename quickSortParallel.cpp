#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"

using namespace std;

void generateMatrix(int size, string name)
{
	int range = 10000;
	ofstream out;
	out.open(name);
	out << size << endl;
	for (auto i = 0; i < size; i++)
		out << rand() % range << " ";
	out.close();
	return;
}

int comparator(const void* p1, const void* p2)
{
	const int* x = (int*)p1;
	const int* y = (int*)p2;

	if (*x > *y)
		return 1;
	if (*x < *y)
		return -1;
	return 0;
}

int getLen(const string name)
{
	int len;
	ifstream in;

	in.open(name);
	if (!in.is_open()) throw std::runtime_error("error open file " + name); 
	in >> len;
	in.close();
	return len;
}

void readFile(const string name, int* array)
{
	int len;
	ifstream in;
	in.open(name);
	if (!in.is_open()) throw std::runtime_error("error open file " + name);
	
	in >> len;
	for (auto i = 0; i < len; i++)
		in >> array[i];


	in.close();
}


void intersation_sort(int *mas, int count, bool(*comp)(int, int)) {
	int key = 0;
	int temp = 0;
	for (int i = 0; i < count - 1; i++)
	{
		key = i + 1;
		temp = mas[key];
		for (int j = i + 1; j > 0; j--)
		{
			if (comp(temp, mas[j - 1]))
			{
				mas[j] = mas[j - 1];
				key = j - 1;
			}
		}
		mas[key] = temp;
	}
}

void MPIQsort(const int size, const int rank, const int len, int* array, int* lengths)
{
	auto iterations = log(size) / log(2);
	int* subArray = new int[len];
	int* buffer = new int[len];
	int* offsets = new int[size] {0};

	for (auto iteration = 0; iteration < iterations; iteration++)
	{
		// count offsets for Scatterv
		for (auto i = 1; i < size; i++)
			offsets[i] = offsets[i - 1] + lengths[i - 1];

		int mySize;
		MPI_Scatter(lengths, 1, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(array, lengths, offsets, MPI_INT, subArray, mySize, MPI_INT, 0, MPI_COMM_WORLD);

		//  communicators split
		int color = rank / pow(2, iterations - iteration);
		MPI_Comm MPI_LOCAL_COMMUNICATOR;
		MPI_Comm_split(MPI_COMM_WORLD, color, rank, &MPI_LOCAL_COMMUNICATOR);

	
		int localRank, localSize;
		MPI_Comm_rank(MPI_LOCAL_COMMUNICATOR, &localRank);
		MPI_Comm_size(MPI_LOCAL_COMMUNICATOR, &localSize);

		//find pivot
		auto pivot = 0;
		int* medianArray = new int[3];
		if (localRank == 0 && mySize != 0)
		{
			medianArray[0] = subArray[0];
			medianArray[1] = subArray[mySize];
			medianArray[2] = subArray[mySize / 2];
		}

		auto less = [](int lh, int rh) { return lh < rh; };
		intersation_sort(medianArray, 3, less);
		
		pivot = medianArray[1];

		//broadcast pivot
		MPI_Bcast(&pivot, 1, MPI_INT, 0, MPI_LOCAL_COMMUNICATOR);

		// now do in-place partition
		auto lessSize = 0;
		auto greaterSize = 0;
		auto lessBorder = 0;
		auto greaterBorder = lengths[rank] - 1;
		while (lessBorder <= greaterBorder)
		{
			if (subArray[lessBorder] <= pivot)
			{
				lessBorder++;
				lessSize++;
			}
			else if (subArray[greaterBorder] > pivot)
			{
				greaterBorder--;
				greaterSize++;
			}
			else
			{
				auto tmp = subArray[lessBorder];
				subArray[lessBorder] = subArray[greaterBorder];
				subArray[greaterBorder] = tmp;
			}
		}

		// now check if localRank is from "upper" or "lower" group and send/receive
		// from "lower" - send greater to partner's buffer
		// from "upper" - send less to partner's buffer
		auto rankFromLowerGroup = localRank < localSize / 2;
		auto sendTo = 0;
		auto recFrom = 0;
		auto bufferSize = 0;
		if (rankFromLowerGroup)
		{
			sendTo = recFrom = localRank + localSize / 2;
			MPI_Send(&greaterSize, 1, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
			MPI_Recv(&bufferSize, 1, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
			MPI_Send(subArray + lessSize, greaterSize, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
			MPI_Recv(buffer, bufferSize, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
		}
		else
		{
			sendTo = recFrom = localRank - localSize / 2;
			MPI_Recv(&bufferSize, 1, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
			MPI_Send(&lessSize, 1, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
			MPI_Recv(buffer, bufferSize, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
			MPI_Send(subArray, lessSize, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
		}

		// now merge arrays
		if (rankFromLowerGroup)
		{
			memcpy(buffer + bufferSize, subArray, lessSize * sizeof(int));
			bufferSize += lessSize;
		}
		else
		{
			memcpy(buffer + bufferSize, subArray + lessSize, greaterSize * sizeof(int));
			bufferSize += greaterSize;
		}

		// if last iteration - do sequential qsort
		if (iteration == iterations - 1)
			qsort(buffer, bufferSize, sizeof(int), comparator);

		// update lengths
		MPI_Allgather(&bufferSize, 1, MPI_INT, lengths, 1, MPI_INT, MPI_COMM_WORLD);

		// recalculate offsets for Gatherv
		for (auto i = 1; i < size; i++)
			offsets[i] = offsets[i - 1] + lengths[i - 1];
		MPI_Gatherv(buffer, bufferSize, MPI_INT, array, lengths, offsets, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Comm_free(&MPI_LOCAL_COMMUNICATOR);
	}

	delete[] subArray;
	delete[] buffer;
	delete[] offsets;
}

void printResult(const string name, const int len, const int* array)
{
	ofstream out;
	out.open(name);

	out << len << endl;
	for (auto i = 0; i < len; i++)
		out << array[i] << " ";
	out.close();
}


bool FileIsExist(std::string filePath)
{
	bool isExist = false;
	std::ifstream fin(filePath.c_str());

	if (fin.is_open())
		isExist = true;

	fin.close();
	return isExist;
}

int main(int argc, char* argv[])
{
	try
	{
		MPI_Init(&argc, &argv);
		int size, rank;

		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (rank == 0 && size % 2 && size != 1)
		{
			cerr << "Required size equals power of 2";
			MPI_Abort(MPI_COMM_WORLD, 0);
			return 0;
		}

		int* array = {};

		string name;
		long sizeArray;
		if (rank == 0)
		{
			cout << "Input file name" << endl;
			cin >> name;
			name = name + ".txt";

			if(!FileIsExist(name))
			{
				cout << "Create New File->\n Enter size sortArray:" << endl;
				cin >> sizeArray;
				generateMatrix(sizeArray, name);
			}
			else
			{
				cout << "file exist->read file" << endl;
				sizeArray = getLen(name);
			}
			
			array = new int[sizeArray];
			readFile(name, array);
			cout << "Output file name" << endl;
			cin >> name;
			name = name + ".txt";
		}

		MPI_Bcast(&sizeArray, 1, MPI_LONG, 0, MPI_COMM_WORLD);

		// create lengths array outa qsort to calculate clear MPI time
		int* lengths = new int[size];
		for (auto i = 0; i < size; i++)
			lengths[i] = sizeArray / size;
		for (auto i = 0; i < sizeArray % size; i++)
			lengths[i]++;

		// qsort
		const auto start = MPI_Wtime();
		if (size == 1)
			qsort(array, sizeArray, sizeof(int), comparator);
		else
			MPIQsort(size, rank, sizeArray, array, lengths);
		const auto finish = MPI_Wtime();

		if (rank == 0)
		{
			cout <<" SizeArray: " << sizeArray << " Time: " << finish - start << endl;
			printResult(name, sizeArray, array);
		}

		delete[] array;
		delete[] lengths;

		MPI_Finalize();
	}
	catch (exception e)
	{
		cout << e.what();
	}

	return 0;
}