#include <igl/read_triangle_mesh.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/Timer.h>
// #include <igl/embree/unproject_onto_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <iostream>
#include <vector>

#include <SSP_decimate.h>
#include <query_coarse_to_fine.h>
#include <single_collapse_data.h>

typedef std::tuple<Eigen::Matrix4f, Eigen::Matrix4f, Eigen::Vector4f>
    camera_info;

std::tuple<std::vector<std::vector<int>>,
           std::vector<std::vector<Eigen::Vector3d>>>
get_pt_mat(camera_info cam, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
           int W, int H) {
  auto view = std::get<0>(cam);
  auto proj = std::get<1>(cam);
  auto vp = std::get<2>(cam);

  // TODO: implement this function
  return std::make_tuple(std::vector<std::vector<int>>(),
                         std::vector<std::vector<Eigen::Vector3d>>());
};

int main(int argc, char *argv[]) {
  using namespace Eigen;
  using namespace std;

  // load mesh
  MatrixXd VO, V;
  MatrixXi FO, F;
  {
    std::string model_name = argc > 1 ? argv[1] : "bunny";
    igl::read_triangle_mesh("../../meshes/" + model_name + ".obj", VO, FO);
    cout << "original mesh: |V| " << VO.rows() << ", |F|: " << FO.rows()
         << endl;
  }

  // decimate the input mesh using SSP
  SparseMatrix<double> P;
  int tarF = 1000;  // target number of faces
  int dec_type = 0; // decimation type (0:qslim, 1:midpoint, 2:vertex removal)
  VectorXi IM, FIM;
  vector<single_collapse_data> decInfo;
  vector<vector<int>> decIM;
  VectorXi IMF;

  igl::Timer timer;
  timer.start();
  SSP_decimate(VO, FO, tarF, dec_type, V, F, IMF, IM, decInfo, decIM, FIM);
  timer.stop();
  cout << "decimation time: " << timer.getElapsedTime() << " s" << endl;
  cout << "decimated mesh: |V| " << V.rows() << ", |F|: " << F.rows() << endl;

  // get barycentric coordinates on the coarse mesh for querying (I set it to
  // the coarse vertices in this case)
  MatrixXd BC(V.rows(), 3);
  BC.setZero();
  MatrixXi BF(V.rows(), 3);
  BF.setZero();
  VectorXi FIdx(V.rows());
  FIdx.setZero();
  for (int fIdx = 0; fIdx < F.rows(); fIdx++) {
    for (int ii = 0; ii < F.cols(); ii++) {
      int vIdx = F(fIdx, ii);
      if (BC.row(vIdx).sum() == 0.0) {
        BC(vIdx, ii) = 1;
        BF.row(vIdx) = F.row(fIdx);
        FIdx(vIdx) = fIdx;
      }
    }
  }

  // query coarse to fine
  timer.start();
  query_coarse_to_fine(decInfo, IM, decIM, IMF, BC, BF, FIdx);
  timer.stop();
  cout << "query time: " << timer.getElapsedTime() << " s" << endl;

  // funciton to generate the picture
  auto writePNG = [&](const std::string &name, int W, int H,
                      Eigen::Matrix4f view, Eigen::Matrix4f proj,
                      Eigen::Vector4f vp) {
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, A;
    igl::png::writePNG(R, G, B, A, name);
  };

  // compute point location on the fine mesh
  MatrixXd pt(BC.rows(), 3);
  pt.setZero();
  for (int ii = 0; ii < BC.rows(); ii++) {
    pt.row(ii) = BC(ii, 0) * VO.row(BF(ii, 0)) + BC(ii, 1) * VO.row(BF(ii, 1)) +
                 BC(ii, 2) * VO.row(BF(ii, 2));
  }

  // visualize the prolongation operator
  igl::opengl::glfw::Viewer viewer;
  Vector4f backColor;
  backColor << 208 / 255., 237 / 255., 227 / 255., 1.;
  viewer.core().background_color = backColor;
  viewer.data().set_mesh(VO, FO);
  const Eigen::RowVector3d blue(149.0 / 255, 217.0 / 255, 244.0 / 255);
  viewer.data().set_colors(blue);
  viewer.data().add_points(pt, Eigen::RowVector3d(0, 0, 0));
  viewer.data().point_size = 10;

  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer,
                                 unsigned char key, int mod) -> bool {
    switch (key) {
    case '0':
      viewer.data().clear();
      viewer.data().set_mesh(VO, FO);
      viewer.data().set_colors(blue);
      viewer.data().add_points(pt, Eigen::RowVector3d(0, 0, 0));
      viewer.data().point_size = 10;
      break;
    case '1':
      viewer.data().clear();
      viewer.data().set_mesh(V, F);
      viewer.data().set_colors(blue);
      break;
    default:
      return false;
    }
    return true;
  };
  viewer.launch();
}
