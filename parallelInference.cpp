#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <future>

#define TorchAvailable true



#if TorchAvailable
#include <torch/torch.h>
struct LayerModelImpl : torch::nn::Module {
	LayerModelImpl(int64_t N, int64_t M)
		:linear1(N, 100), linear2(100, 50), linear3(50, M)
	{
		register_module("linear1", linear1);
		register_module("linear2", linear2);
		register_module("linear3", linear3);
	}
	torch::Tensor forward(torch::Tensor input) {

		return torch::relu(linear3(torch::relu(linear2(torch::relu(linear1(input))))));
	}
	torch::nn::Linear linear1, linear2, linear3;

};
TORCH_MODULE(LayerModel);
#endif


class Model
{
public:
	size_t n, m;
	Model(size_t n, size_t m) :n{ n }, m{ m }
	{
#ifdef TorchAvailable
		model = LayerModel(n, m);
#endif
	}
#ifdef TorchAvailable
	LayerModel model =nullptr;
#endif
	size_t GetAction(std::vector<float> & feats)
	{
#ifdef TorchAvailable
		torch::Tensor features = torch::empty({ 1,(long)(n) });
		if (feats.size() != n)
		{
			throw "unsuitable dimension";
		}
		for (size_t i = 0; i < n; i++)
		{
			features[0][i] = feats[i];
		}
		//Note that NoGradGuard was called elsewhere
		torch::Tensor prediction = model->forward(features);
		//Convert to c++ style vector. 
		std::vector<float> logprobs(prediction.data_ptr<float>(), prediction.data_ptr<float>() + prediction.numel());
		float highest = std::numeric_limits<float>::min();

		size_t prescribedAction = 0;
		for (size_t action = 0; action < m;action++)
		{
			if (logprobs[action] > highest)
			{
				highest = logprobs[action];
				prescribedAction = action;
			}
		}
		return prescribedAction;

#else
		std::this_thread::sleep_for(std::chrono::microseconds(200));

		return 1;
#endif

	}

};




class Worker
{
public:
	Model* model;
	size_t toCollect;
	std::mt19937 gen;
	std::uniform_real_distribution<float> dist;

	Worker(Model* model, size_t toCollect,size_t seed) :model{ model }, toCollect{toCollect},gen(seed), dist(0.0,1.0)
	{

	}
	size_t result;
	void doWork(int batchnumber)
	{
#if TorchAvailable
		torch::NoGradGuard guard;
#endif
		auto time1 = std::chrono::high_resolution_clock::now();
		result = 0;
		float init = 0.0f;
		size_t n = model->n;
		std::vector<float> feats(n,init );
		
		for (size_t i = 0; i < toCollect; i++)
		{
			for (size_t j = 0; j < n; j++)
			{
				feats[j] = dist(gen);
				auto action = model->GetAction(feats);
				//Actual business case involves result being a more complex function of 
				//actions, i.e. reinforcement learning scenario. I keep it simple here:
				result += action;
			}					
		}
		auto time2 = std::chrono::high_resolution_clock::now();
		std::cout << "Completed batch: " << batchnumber << "  timing: " << ((time2 - time1).count() / 1000000) << std::endl;
	}

};



int main()
{
	size_t n = 10, m = 20;
	Model model(n, m);
	unsigned concurrentThreadsSupported = std::thread::hardware_concurrency();
	std::cout << "Threads detected: " << concurrentThreadsSupported << std::endl;

	unsigned batches = concurrentThreadsSupported;

	std::vector<Worker> workers;
	workers.reserve(batches);
	{
		for (size_t i = 0; i < batches; i++)
		{
			workers.push_back(Worker(&model, 100, 111 * i));
		}
	}
	std::cout << "parallel execution: " << std::endl;
	std::vector<std::future<void>> futures;
	futures.reserve(batches);
	for (size_t i = 0; i < batches; i++)
	{
		futures.push_back(std::async([](Worker* worker, int number) {worker->doWork(number);}, &(workers[i]), i));
	}
	for (auto &e : futures)
	{//wait until all parallel computations are completed
		e.get();
	}
	std::cout << "serial execution: " << std::endl;
	size_t number{ 0 };
	for (auto&worker : workers)
	{
		worker.doWork(number++);
	}
	return 0;
}