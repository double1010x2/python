#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tqdm.hpp"

using namespace std;
namespace py = pybind11;

#define DBL_MAX     std::numeric_limits<double>::max()
#define rSOM_ijk    ptr_result[i*N*C+j*C+k] 
#define SOM_ijk     som[i*N*C+j*C+k] 
#define pSOM_ijk    ptr_som[i*N*C+j*N+k] 
#define ij0         ij.first
#define ij1         ij.second


// get index 
auto find_BMU(double* som, int const M, int const N,
                       double* data, int const C) -> pair<int,int>
{

    double distSq = 0.;
    auto ij = make_pair(0, 0);
    double fmax = DBL_MAX;

    for (int i=0; i<M; ++i){
        for (int j=0; j<N; ++j){
            distSq = 0.;
            for (int k=0; k<C; ++k){
                distSq += pow(data[k]-SOM_ijk, 2); 
            }
        //    printf("1: i(%d), j(%d), dist(%f), fmax(%f)\n", i, j, distSq, fmax);
            if (distSq < fmax){
                ij.first  = i;
                ij.second = j;
                fmax = distSq;
            }
//            printf("2: i(%d), j(%d), dist(%f), fmax(%f)\n", i, j, distSq, fmax);
        }
    }
    return ij;
}

auto find_BMU_cpp(py::array_t<double> som, py::array_t<double> data) -> py::array_t<int>
{
  py::buffer_info buf_som   = som.request();
  py::buffer_info buf_data  = data.request();

  auto ptr_som  = static_cast<double*>(buf_som.ptr);
  auto ptr_data = static_cast<double*>(buf_data.ptr);

  int const M    = buf_som.shape[0];
  int const N    = buf_som.shape[1];
  int const C    = buf_som.shape[2];

  //printf("[DBG-MNC]: M(%d), N(%d), C(%d)\n", M, N, C);
  auto result = py::array_t<int>(2);
  auto buf_result = result.request();
  auto ptr_result = static_cast<int*>(buf_result.ptr);

  auto ij = find_BMU(ptr_som, M, N, ptr_data, C);
  ptr_result[0] = ij0;
  ptr_result[1] = ij1;

  result.resize({2});

  return result;
}

auto find_BMU_image(py::array_t<double> som, py::array_t<double> data) -> py::array_t<int>
{
  py::buffer_info buf_som   = som.request();
  py::buffer_info buf_data  = data.request();

  int const n_data   = buf_data.shape[0];
  int const M        = buf_som.shape[0];
  int const N        = buf_som.shape[1];
  int const C        = buf_som.shape[2];
  /***
  cout << "n_data = " << n_data << endl;
  cout << "C= " << C << endl;
  cout << "M= " << M << endl;
  cout << "N= " << N << endl;
  ***/
  py::array_t<int> result = py::array_t<int>(M*N);
  py::buffer_info buf_result = result.request();

  auto ptr_som  = static_cast<double*>(buf_som.ptr);
  auto ptr_data = static_cast<double*>(buf_data.ptr);
  auto ptr_result = static_cast<int*>(buf_result.ptr);

  // initial all result
  for (int i=0; i<M*N; ++i)
    ptr_result[i] = 0;

  tqdm bar;
  pair<int, int> ij;
  for (int data_i=0; data_i<n_data; ++data_i){
    ij = find_BMU(ptr_som, M, N, ptr_data+data_i*C, C);
    ptr_result[ij0*N+ij1] += 1;
    bar.progress(static_cast<int>(data_i), n_data);
  }
  bar.finish();

  result.resize({M,N});
  return result;
}

auto dbgMatrix(py::array_t<double> som) -> void
{
  py::buffer_info buf_som   = som.request();
  auto ptr_som  = static_cast<double*>(buf_som.ptr);

  int const M = buf_som.shape[0];
  int const N = buf_som.shape[1];
  int const C = buf_som.shape[2];

  for (int i=0; i<(M*N*C); ++i)
    printf("i(%d), val(%f)\n", i, ptr_som[i]);
}


auto clustering(
                    py::array_t<double> som,
                    py::array_t<double> data,
                    double const learn_rate,
                    double const radius_sq,
                    int const step) -> py::array_t<double>  
{
  py::buffer_info buf_som   = som.request();
  py::buffer_info buf_data  = data.request();

  int const dbg = 0;
  auto ptr_som  = static_cast<double*>(buf_som.ptr);
  auto ptr_data = static_cast<double*>(buf_data.ptr);

  int const n_data   = buf_data.shape[0];
  int const C        = buf_data.shape[1];
  int const M        = buf_som.shape[0];
  int const N        = buf_som.shape[1];

  // Return SOM results
  auto result = py::array_t<double>(M*N*C);
  auto buf_result = result.request();
  auto ptr_result = static_cast<double*>(buf_result.ptr);
  memcpy(ptr_result, ptr_som, sizeof(double) * M*N*C);

  if (dbg) {
    double som_sum = 0;
    for (int i=0; i<M*N*C; ++i) som_sum += ptr_result[i];
    printf("(n_data, M, N, C, som_sum) = (%d,%d,%d,%d,%f)\n", n_data, M, N, C, som_sum);
  }


  int imin = 0;
  int imax = 0;
  int jmax = 0;
  int jmin = 0;
  double distSq = 0;
  double* data_sub;
  //vector<int> ij;
  auto ij = make_pair(0, 0);
  tqdm bar;
  for (int data_i=0; data_i<n_data; ++data_i){

    data_sub = ptr_data + data_i*C;
    ij = find_BMU(ptr_result, M, N, data_sub, C);

    if (radius_sq < 1e-3){
        for (int i=0; i<M; ++i){
            for (int j=0; j<N; ++j){
                for (int k=0; k<C; ++k){
                    if (dbg) cout << "[Before]: " << rSOM_ijk;
                    rSOM_ijk += learn_rate * (data_sub[k]-rSOM_ijk);
                    if (dbg)  cout << ", [After]: " << rSOM_ijk << endl;
                }
            }
        }
    } else {

      imin = max(0,  ij0-step);
      imax = min(M,  ij0+step+1); // +1 for symmetric
      jmin = max(0,  ij1-step);   
      jmax = min(N,  ij1+step+1);   // +1 for symmetric
      if (dbg){
        printf("[DBG-ij]-data(%d),i(%d),j(%d),imin(%d),imax(%d),jmin(%d),jmax(%d)\n", data_i,ij0,ij1,imin,imax,jmin,jmax);
      } 
      for (int i=imin; i<imax; ++i){
          for (int j=jmin; j<jmax; ++j){
              distSq = static_cast<double>(pow(i-ij0, 2)+pow(j-ij1, 2));
              distSq = exp(-distSq * 0.5 / radius_sq);
              if (dbg) printf("i(%d),j(%d),C(%d), dist(%f), radius(%f)\n", i, j, C, distSq, radius_sq);
              for (int k=0; k<C; ++k){
                  if (dbg) cout << "[Before]: " << rSOM_ijk;
                  rSOM_ijk += learn_rate * distSq * (data_sub[k]-rSOM_ijk);
                  if (dbg) cout << ", [After]: " << rSOM_ijk << endl;
              }
          }
      }
    }
      bar.progress(static_cast<int>(data_i), n_data);
  }
  bar.finish();

  result.resize({M,N,C});
  return result;
}

PYBIND11_MODULE(SOM, m) {
        m.doc() = "Clustering by Self-Organized Map using pybind11"; // optional module docstring
        m.def("clustering",     &clustering,    "Clustering by Self-Organized");
        m.def("find_BMU_cpp",   &find_BMU_cpp,  "find BMU");
        m.def("find_BMU_image", &find_BMU_image,"return BMU historgram map");
        m.def("dbgMatrix",      &dbgMatrix      ,"debug Matrix order");
}