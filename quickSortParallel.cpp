#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"

using namespace std;

int rankProc;
int sizeProc;

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

int GetPart(int* array, int left_index, int right_index) {
	int pivot = array[left_index];
	int i = left_index - 1;
	int j = right_index + 1;
	while (true) {
		do {
			i++;
		} while (array[i] < pivot);
		do {
			j--;
		} while (array[j] > pivot);

		if (i >= j) {
			return j;
		}
		array[i] += array[j];
		array[j] = array[i] - array[j];
		array[i] -= array[j];
	}
	return 0;
}

void QuickSortLinear(int* array, int left_index, int right_index) {
	if (left_index < right_index)
	{
		int pivot = GetPart(array, left_index, right_index);
		QuickSortLinear(array, left_index, pivot);
		QuickSortLinear(array, pivot + 1, right_index);
	}
}

int GetPartParallel(int* array, int left_index, int right_index, int pivot) {
	int j = -1;
	int tmp;
	for (std::size_t i = left_index; i < right_index + 1; ++i) {
		//printf( "i: %d \tarr[i]: %lf \t x: %lf \t", i, arr[i], x );
		if (array[i] <= pivot) {
			//printf("true");
			j++;
			//printf("\t\tarr[i]: %lf \tarr[j]: %lf \t\t", arr[i], arr[j]);
			tmp = array[j];
			array[j] = array[i];
			array[i] = tmp;
			//printf("\t\tarr[i]: %lf \tarr[j]: %lf \t\t", arr[i], arr[j]);
		}
		//printf("\n");
	}
	return j;
}

void MPIQsort(int* array, int left_index, int right_index)
{
	int array_length = right_index - left_index + 1;

	int n_stages = (int)(log(sizeProc) / log(2));
	int n_available_procs = (int)(pow(2.0, (double)n_stages));


	int* procs_array_sizes = nullptr;
	int receive_buffer_size;
	if (rankProc == 0) {
		procs_array_sizes = (int*)malloc(sizeProc * sizeof(int));
		receive_buffer_size = array_length / sizeProc;
		for (std::size_t i = 0; i < sizeProc - 1; ++i)
			procs_array_sizes[i] = receive_buffer_size;
		procs_array_sizes[sizeProc - 1] = array_length - (sizeProc - 1) * receive_buffer_size;
	}

	MPI_Scatter(procs_array_sizes, 1, MPI_INT, &receive_buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int* receive_buffer = (int*)malloc(receive_buffer_size * sizeof(int));

	int* procs_array_indices = nullptr;
	if (rankProc == 0) {
		procs_array_indices = (int*)malloc(sizeProc * sizeof(int));
		procs_array_indices[0] = 0;
		for (std::size_t i = 1; i < n_available_procs; ++i)
			procs_array_indices[i] = procs_array_indices[i - 1] + procs_array_sizes[i - 1];
	}

	MPI_Scatterv(array,
		procs_array_sizes,
		procs_array_indices,
		MPI_INT,
		receive_buffer,
		receive_buffer_size,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

	MPI_Request request;
	MPI_Status status;

	//теперь провести log2 стадий
	//if( myID == 0 ){printf("devinded: %d\n", devinded); }


	int partner_procs_rank;
	int send_count;
	int* send_buffer;

	int* cur_recieve_buffer;
	int* new_receive_buffer;
	int cur_receive_count;
	int new_receive_buffer_size;

	int pivot;
	int procs_type;
	for (std::size_t stage_index = 0; stage_index < n_stages; ++stage_index) {
		//разделились на верхние и нижние
		//в type == 0  - меньшие значения
		if (rankProc % n_available_procs < n_available_procs / 2) {
			procs_type = 0;
		}
		else {
			procs_type = 1;
		}

		//надо выяснить опорный элемент
		//выяснять будет наименьший в группе
		if (rankProc % n_available_procs == 0) {
			pivot = 0;
			if (receive_buffer_size != 0) {
				pivot = receive_buffer[0];
			}
			for (std::size_t i = rankProc + 1; i < rankProc + n_available_procs; ++i) {
				MPI_Isend(&pivot,
					1,
					MPI_INT,
					i,
					0,
					MPI_COMM_WORLD,
					&request);
			}
		}
		else {
			int from;
			from = rankProc - rankProc % n_available_procs;
			MPI_Irecv(&pivot,
				1,
				MPI_INT,
				from,
				MPI_ANY_TAG,
				MPI_COMM_WORLD,
				&request);
		}

		MPI_Wait(&request, &status);

		//сделать partition
		int partition = GetPartParallel(receive_buffer, 0, receive_buffer_size - 1, pivot);
		//надо обменять половины
		if (procs_type == 0) {
			//собираю меньшие (левые) части
			//моя пара = +n_available_procs/2
			partner_procs_rank = rankProc + n_available_procs / 2;
			//скажем партнеру, сколько передадим ему
			send_count = receive_buffer_size - partition - 1;

			MPI_Send(&send_count, 1, MPI_INT, partner_procs_rank, 0, MPI_COMM_WORLD);

			//узнаем сколько принимать
			MPI_Recv(&cur_receive_count, 1, MPI_INT, partner_procs_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			//подготовим новый массив побольше
			new_receive_buffer_size = partition + 1 + cur_receive_count;
			new_receive_buffer = (int*)malloc((new_receive_buffer_size) * sizeof(int));
			/*
			if( myID == testID )
				printf( "id: %d \ti: %d \tm: %d \t send_count: %d \tcur_receive_count: %d \tnew_receive_buffer_size: %d\n",
					myID, i, m, send_count, cur_receive_count, new_receive_buffer_size);
			*/

			for (std::size_t j = 0; j < partition + 1; ++j)
				new_receive_buffer[j] = receive_buffer[j];
			/*
			if( myID == testID ) {
				for( j = 0; j < new_receive_buffer_size; ++j ) printf("%lf ", new_receive_buffer[j]);
				printf("\n");
			}
			*/
			//и правильный указатель
			send_buffer = receive_buffer;
			if (partition + 1 < receive_buffer_size)
				send_buffer = &(receive_buffer[partition + 1]);

			cur_recieve_buffer = new_receive_buffer;
			if (cur_receive_count > 0)
				cur_recieve_buffer = &(new_receive_buffer[partition + 1]);


			//sendrecv

			MPI_Isend(send_buffer, send_count, MPI_INT,
				partner_procs_rank, 0,
				MPI_COMM_WORLD, &request);
			MPI_Irecv(cur_recieve_buffer, cur_receive_count, MPI_INT,
				partner_procs_rank, MPI_ANY_TAG,
				MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);

			/*
			if( myID == testID ) {
				for( j = 0; j < new_receive_buffer_size; ++j ) printf("%lf ", new_receive_buffer[j]);
				printf("\n");
			}
			*/

			free(receive_buffer);
			receive_buffer = new_receive_buffer;
			receive_buffer_size = new_receive_buffer_size;


		}
		else {
			//собираю большие, правые части
			//моя пара = -n_available_procs/2
			partner_procs_rank = rankProc - n_available_procs / 2;

			//узнаем сколько принимать
			MPI_Recv(&cur_receive_count, 1, MPI_INT, partner_procs_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			//скажем паре, сколько передади ему
			send_count = partition + 1;
			MPI_Send(&send_count, 1, MPI_INT, partner_procs_rank, 0, MPI_COMM_WORLD);


			//подготовим новый массив побольше
			new_receive_buffer_size = receive_buffer_size - partition - 1 + cur_receive_count;
			new_receive_buffer = (int*)malloc((new_receive_buffer_size) * sizeof(int));
			/*
			if( myID == testID )
				printf( "id: %d \ti: %d \tm: %d \t send_count: %d \tcur_receive_count: %d \tnew_receive_buffer_size: %d\n",
					myID, i, m, send_count, cur_receive_count, new_receive_buffer_size);
			*/

			for (std::size_t j = 0; j < receive_buffer_size - partition - 1; ++j)
				new_receive_buffer[j] = receive_buffer[j + partition + 1];
			/*
			if( myID == testID ) {
				for( j = 0; j < new_receive_buffer_size; ++j ) printf("%lf ", new_receive_buffer[j]);
				printf("\n");
			}
			*/
			//и правильный указатель
			send_buffer = receive_buffer;
			cur_recieve_buffer = new_receive_buffer;
			if (cur_receive_count > 0)
				cur_recieve_buffer = &(new_receive_buffer[receive_buffer_size - partition - 1]);


			//sendrecv
			MPI_Isend(send_buffer, send_count, MPI_INT,
				partner_procs_rank, 0,
				MPI_COMM_WORLD, &request);
			MPI_Irecv(cur_recieve_buffer, cur_receive_count, MPI_INT,
				partner_procs_rank, MPI_ANY_TAG,
				MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);

			/*
			if( myID == testID ) {
				for( j = 0; j < new_receive_buffer_size; ++j ) printf("%lf ", new_receive_buffer[j]);
				printf("\n");
			}
			*/

			free(receive_buffer);
			receive_buffer = new_receive_buffer;
			receive_buffer_size = new_receive_buffer_size;
		}

		n_available_procs = n_available_procs / 2;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	QuickSortLinear(receive_buffer, 0, receive_buffer_size - 1);
	MPI_Gather(&receive_buffer_size, 1, MPI_INT, procs_array_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rankProc == 0) {
		procs_array_indices[0] = 0;
		for (std::size_t i = 1; i < sizeProc; ++i)
			procs_array_indices[i] = procs_array_indices[i - 1] + procs_array_sizes[i - 1];
	}

	MPI_Gatherv(receive_buffer,
		receive_buffer_size,
		MPI_INT, array,
		procs_array_sizes,
		procs_array_indices,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

	if (rankProc == 0) {
		free(procs_array_sizes);
		free(procs_array_indices);
	}

	free(receive_buffer);
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

		MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);
		MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);

		if (rankProc == 0 && sizeProc % 2 && sizeProc != 1)
		{
			cerr << "Required size equals power of 2";
			MPI_Abort(MPI_COMM_WORLD, 0);
			return 0;
		}

		int* array =  {};
		int* array2 ={};
		string name;
		long sizeArray=5;
		long sizeArray123 = 0;
		if (rankProc == 0)
		{
			cout << "Input file name" << endl;
			cin >> name;
			name = name + ".txt";

			if (!FileIsExist(name))
			{
				cout << "Create New File->\n Enter size sortArray:" << endl;
				cin >> sizeArray;
				generateMatrix(sizeArray, name);
			}
			else
			{
				cout << "file exist->read file" << endl;
				sizeArray = getLen(name);
				cout << "Array size= " << sizeArray << std::endl;
			}
			
			array = new int[sizeArray];
			
			readFile(name, array);
			
			cout << "Output file name" << endl;
			cin >> name;
			name = name + ".txt";
		}

		// qsort
		const auto start = MPI_Wtime();
		if (sizeProc == 1)
			qsort(array, sizeArray, sizeof(int), comparator);
		else
			MPIQsort(array, 0, sizeArray -1);
		const auto finish = MPI_Wtime();

		if (rankProc == 0)
		{
			cout <<" SizeArray: " << sizeArray << " Time: " << finish - start << endl;
			printResult(name, sizeArray, array);
		}

		delete[] array;


		MPI_Finalize();
	}
	catch (exception e)
	{
		cout << e.what();
	}



	return 0;
}