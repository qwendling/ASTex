#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ASTex {

/**
 Simple Tiling n blending algo
 Need Gaussian imagea as input
 */
template <typename IMG> class Tiling_n_Blending {
  using PIXT = typename IMG::PixelType;
  using EPIXT = typename IMG::DoublePixelEigen;

  const IMG &img_input_; // exemple d'entrée

  EPIXT img_average_;
  Eigen::Matrix3d F_average_;
  double lattice_resolution_ = 20;

  struct Face {
    Eigen::Vector3i Vertices_3d;
    Eigen::Vector3i Vertices_uv;
    Eigen::Vector3d F_normal;
    Eigen::Matrix3d Deformation_gradient;
    Eigen::Matrix3d Sim;
    double area;
    Face() : area(0.0) { Sim = Eigen::Matrix3d::Identity(); }
  };
  std::vector<Eigen::Vector3d> Vertices_3d;
  std::vector<Eigen::Vector2d> Vertices_uv;
  std::vector<Eigen::Matrix3d> Vertices_deform;
  std::vector<Face> Faces;

  struct kd_tree {
    kd_tree *left;
    kd_tree *right;
    std::vector<Face> f;
    Face f_median;

    kd_tree(const std::vector<Face> &face)
        : left(nullptr), right(nullptr), f(face) {}
    kd_tree() : left(nullptr), right(nullptr) {}
  };

  kd_tree *tree_;

  kd_tree *build_kd_tree(std::vector<Face> &faces,
                         std::vector<Eigen::Vector2d> vertices_uv, int depth) {
    if (faces.empty())
      return nullptr;
    if (faces.size() < 10 || depth > 20)
      return new kd_tree(faces);

    int axis = depth % 2;
    if (axis == 0) {
      std::sort(faces.begin(), faces.end(), [&](const Face &a, const Face &b) {
        Eigen::Vector2d a0 = vertices_uv[a.Vertices_uv(0)];
        Eigen::Vector2d a1 = vertices_uv[a.Vertices_uv(1)];
        Eigen::Vector2d a2 = vertices_uv[a.Vertices_uv(2)];
        Eigen::Vector2d b0 = vertices_uv[b.Vertices_uv(0)];
        Eigen::Vector2d b1 = vertices_uv[b.Vertices_uv(1)];
        Eigen::Vector2d b2 = vertices_uv[b.Vertices_uv(2)];
        return std::min({a0.x(), a1.x(), a2.x()}) <
               std::min({b0.x(), b1.x(), b2.x()});
      });
    } else {
      std::sort(faces.begin(), faces.end(), [&](const Face &a, const Face &b) {
        Eigen::Vector2d a0 = vertices_uv[a.Vertices_uv(0)];
        Eigen::Vector2d a1 = vertices_uv[a.Vertices_uv(1)];
        Eigen::Vector2d a2 = vertices_uv[a.Vertices_uv(2)];
        Eigen::Vector2d b0 = vertices_uv[b.Vertices_uv(0)];
        Eigen::Vector2d b1 = vertices_uv[b.Vertices_uv(1)];
        Eigen::Vector2d b2 = vertices_uv[b.Vertices_uv(2)];
        return std::min({a0.y(), a1.y(), a2.y()}) <
               std::min({b0.y(), b1.y(), b2.y()});
      });
    }
    int median = faces.size() / 2;
    kd_tree *node = new kd_tree();
    node->f_median = faces[median];

    std::vector<Face> leftFaces;
    std::vector<Face> rightFaces;
    Eigen::Vector2d m0 = vertices_uv[faces[median].Vertices_uv(0)];
    Eigen::Vector2d m1 = vertices_uv[faces[median].Vertices_uv(1)];
    Eigen::Vector2d m2 = vertices_uv[faces[median].Vertices_uv(2)];
    double median_axis = std::min({m0[axis], m1[axis], m2[axis]});
    /*for(int i=0;i<=median;i++){
            leftFaces.push_back(faces[i]);
    }
    for(int i=median;i<faces.size();i++){
            rightFaces.push_back(faces[i]);
    }*/

    for (auto &f : faces) {
      Eigen::Vector2d m0 = vertices_uv[f.Vertices_uv(0)];
      Eigen::Vector2d m1 = vertices_uv[f.Vertices_uv(1)];
      Eigen::Vector2d m2 = vertices_uv[f.Vertices_uv(2)];
      if (std::min({m0[axis], m1[axis], m2[axis]}) <= median_axis) {
        leftFaces.push_back(f);
      }
      if (std::max({m0[axis], m1[axis], m2[axis]}) >= median_axis) {
        rightFaces.push_back(f);
      }
    }
    node->left = build_kd_tree(leftFaces, vertices_uv, depth + 1);
    node->right = build_kd_tree(rightFaces, vertices_uv, depth + 1);

    return node;
  }

  kd_tree *build_kd_tree(std::vector<Face> &faces,
                         std::vector<Eigen::Vector2d> vertices_uv) {
    return build_kd_tree(faces, vertices_uv, 0);
  }

  Face findTriangleContainingPoint(kd_tree *node,
                                   std::vector<Eigen::Vector2d> &Vertices_uv,
                                   const Eigen::Vector2d &p, int depth) {
    if (node == nullptr) {
      return Face();
    }
    if (!node->f.empty()) {
      Face result;
      for (auto &f : node->f) {
        Eigen::Vector2d v0 = 1000 * Vertices_uv[f.Vertices_uv(0)];
        Eigen::Vector2d v1 = 1000 * Vertices_uv[f.Vertices_uv(1)];
        Eigen::Vector2d v2 = 1000 * Vertices_uv[f.Vertices_uv(2)];
        Eigen::Vector2d p2 = 1000 * p;

        double d1 =
            (p2 - v0).x() * (v1 - v0).y() - (v1 - v0).x() * (p2 - v0).y();
        double d2 =
            (p2 - v1).x() * (v2 - v1).y() - (v2 - v1).x() * (p2 - v1).y();
        double d3 =
            (p2 - v2).x() * (v0 - v2).y() - (v0 - v2).x() * (p2 - v2).y();

        int s1 = __signbit(d1) - 1;
        int s2 = __signbit(d2) - 1;
        int s3 = __signbit(d3) - 1;
        /*const double esp = 1e-12;
        if (abs(d1) < esp) {
          d1 = 0.;
        }
        if (abs(d2) < esp) {
          d1 = 0.;
        }
        if (abs(d3) < esp) {
          d1 = 0.;
        }*/

        if (s1 * s2 >= 0 && s2 * s3 >= 0) {
          result = f;
          f.area = 1.;
          break;
        }
      }

      return result;
    }

    int axis = depth % 2;
    Eigen::Vector2d v0 = Vertices_uv[node->f_median.Vertices_uv(0)];
    Eigen::Vector2d v1 = Vertices_uv[node->f_median.Vertices_uv(1)];
    Eigen::Vector2d v2 = Vertices_uv[node->f_median.Vertices_uv(2)];

    if (axis == 0 && p.x() < std::min(v0.x(), std::min(v1.x(), v2.x()))) {
      return findTriangleContainingPoint(node->left, Vertices_uv, p, depth + 1);
    } else if (axis == 0) {
      return findTriangleContainingPoint(node->right, Vertices_uv, p,
                                         depth + 1);
    } else if (axis == 1 &&
               p.y() < std::min(v0.y(), std::min(v1.y(), v2.y()))) {
      return findTriangleContainingPoint(node->left, Vertices_uv, p, depth + 1);
    } else {
      return findTriangleContainingPoint(node->right, Vertices_uv, p,
                                         depth + 1);
    }
  }

#define NORMALIZE_TERM (21. / (2. * 3.1415))
  float Kernel_W(float dist, float h) {
    float q = dist / h;
    if (q > 1.)
      return 0.;
    float m1 = (1.0f - q);
    float m2 = (4.0f * q + 1.0f);
    float h3 = h * h * h;
    float alpha_d = NORMALIZE_TERM * (1. / h3);

    return alpha_d * m1 * m1 * m1 * m1 * m2;
  }

  float kernel_Normalize(float dist, float h) {
    return Kernel_W(dist, h) / Kernel_W(0., h);
  }

protected:
  inline void set_zero(Eigen::Vector2d &v) { v.setZero(); }

  inline void set_zero(Eigen::Vector3d &v) { v.setZero(); }

  inline void set_zero(Eigen::Vector4d &v) { v.setZero(); }

  inline void set_zero(double &v) { v = 0.0; }

  inline void clamp_channel(double &c) {
    if (std::is_floating_point<typename IMG::DataType>::value)
      c = std::max(0.0, std::min(1.0, c));
    else
      c = std::max(0.0, std::min(255.0, c));
  }

  inline void clamp(Eigen::Vector2d &v) {
    clamp_channel(v[0]);
    clamp_channel(v[1]);
  }

  inline void clamp(Eigen::Vector3d &v) {
    clamp_channel(v[0]);
    clamp_channel(v[1]);
    clamp_channel(v[2]);
  }

  inline void clamp(Eigen::Vector4d &v) {
    clamp_channel(v[0]);
    clamp_channel(v[1]);
    clamp_channel(v[2]);
    clamp_channel(v[3]);
  }

  inline void clamp(double &v) { clamp_channel(v); }

  void TriangleGrid(const Eigen::Vector2d &p_uv, Eigen::Vector3d &Bi,
                    Eigen::Vector2i &vertex1, Eigen::Vector2i &vertex2,
                    Eigen::Vector2i &vertex3) {
    const Eigen::Vector2d uv =
        p_uv * 2.0 * std::sqrt(3.0) * lattice_resolution_; // iGridSize

    Eigen::Matrix2d gridToSkewedGrid;
    gridToSkewedGrid << 1.0, -1. / sqrt(3.), 0.0, 2. / sqrt(3.);

    Eigen::Vector2d skewedCoord = gridToSkewedGrid * uv;
    Eigen::Vector2d baseId{std::floor(skewedCoord[0]),
                           std::floor(skewedCoord[1])};
    Eigen::Vector3d temp{skewedCoord[0] - baseId[0], skewedCoord[1] - baseId[1],
                         0.0};
    temp[2] = 1.0 - temp[0] - temp[1];

    if (temp[2] > 0.0) {
      Bi = Eigen::Vector3d(temp[2], temp[1], temp[0]);
      Eigen::Vector2i ibaseId = baseId.cast<int>();
      vertex1 = ibaseId;
      vertex2 = ibaseId + Eigen::Vector2i(0, 1);
      vertex3 = ibaseId + Eigen::Vector2i(1, 0);
    } else {
      Bi = Eigen::Vector3d(-temp[2], 1.0 - temp[1], 1.0 - temp[0]);
      Eigen::Vector2i ibaseId = baseId.cast<int>();
      vertex1 = ibaseId + Eigen::Vector2i(1, 1);
      vertex2 = ibaseId + Eigen::Vector2i(1, 0);
      vertex3 = ibaseId + Eigen::Vector2i(0, 1);
    }
  }

  Eigen::Vector2d MakeCenterUV(Eigen::Vector2i &vertex) {
    Eigen::Matrix2d invSkewMat;
    invSkewMat << 1.0, 0.5, 0.0, sqrt(3.) / 2.;

    Eigen::Vector2d center = (invSkewMat * vertex.cast<double>()) /
                             (2.0 * std::sqrt(3.0) * lattice_resolution_);
    return center;
  }

  Eigen::Matrix2d TransfoMatrix(double theta, double scale) {
    Eigen::Matrix2d Rotation;
    Eigen::Matrix2d Scale;

    double angle = std::fmod(theta, 2. * M_PI);
    angle -= M_PI;

    double cos = std::cos(angle);
    double sin = std::sin(angle);

    Rotation << cos, sin, -sin, cos;
    Scale << scale, 0., 0., scale;

    return Rotation * Scale;
  }

  // original hash version
  Eigen::Vector2d hash(const Eigen::Vector2i &p) {
    Eigen::Matrix2d hashMat;
    hashMat << 127.1, 269.5, 311.7, 183.3;

    Eigen::Vector2d q = hashMat * p.cast<double>();
    q[0] = std::sin(q[0]);
    q[1] = std::sin(q[1]);
    q *= 43758.5453;
    return Eigen::Vector2d(q[0] - std::floor(q[0]), q[1] - std::floor(q[1]));
  }

public:
  Tiling_n_Blending(const IMG &in, const std::string objfile)
      : // constructeur : récupère l'image d'entrée et calcul la moyenne
        img_input_(in) {
    read_obj(objfile);
    tree_ = build_kd_tree(Faces, Vertices_uv);
    // compute average img value
    EPIXT sum;
    set_zero(sum);
    img_input_.for_all_pixels([&](const PIXT &P) {
      EPIXT lv = eigenPixel<double>(P);
      sum += lv;
    });

    F_average_ = Eigen::Matrix3d::Zero();
    for (auto f : Faces) {
      F_average_ += f.Sim;
    }
    F_average_ /= Faces.size();

    img_average_ = sum / double(img_input_.width() * img_input_.height());
  }

  void barycentric(Face &f, Eigen::Vector2d p, double &u, double &v,
                   double &w) {
    Eigen::Vector2d a = Vertices_uv[f.Vertices_uv(0)];
    Eigen::Vector2d b = Vertices_uv[f.Vertices_uv(1)];
    Eigen::Vector2d c = Vertices_uv[f.Vertices_uv(2)];

    Eigen::Vector2d v0 = b - a;
    Eigen::Vector2d v1 = c - a;
    Eigen::Vector2d v2 = p - a;

    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
  }

  int read_obj(const std::string &obj_file) {

    std::ifstream objFile(obj_file);
    if (!objFile) {
      std::cerr << "Erreur : Impossible d'ouvrir le fichier OBJ." << std::endl;
      return 1;
    }
    std::string line;
    while (std::getline(objFile, line)) {
      std::istringstream iss(line);
      std::string token;
      iss >> token;

      if (token == "v") { // Lire les coordonnées des sommets
        Eigen::Vector3d vertex;
        iss >> vertex(0) >> vertex(1) >> vertex(2);
        Vertices_3d.push_back(vertex);
      } else if (token == "vt") { // Lire les coordonnées UV
        Eigen::Vector2d vertex;
        iss >> vertex(0) >> vertex(1);
        Vertices_uv.push_back(vertex);
      } else if (token == "f") {
        Face f;
        for (int i = 0; i < 3; i++) {
          std::string Vertex_info;
          iss >> Vertex_info;
          std::istringstream viss(Vertex_info);
          viss >> f.Vertices_3d(i);
          f.Vertices_3d(i)--;
          viss.ignore();
          viss >> f.Vertices_uv(i);
          f.Vertices_uv(i)--;
        }
        Eigen::Vector3d v0 = Vertices_3d[f.Vertices_3d(0)];
        Eigen::Vector3d v1 = Vertices_3d[f.Vertices_3d(1)];
        Eigen::Vector3d v2 = Vertices_3d[f.Vertices_3d(2)];
        f.F_normal = (v1 - v0).cross(v2 - v0).normalized();
        Faces.push_back(f);
      }
    }

    objFile.close();
    for (auto &f : Faces) {
      Eigen::Vector3d v0 = Vertices_3d[f.Vertices_3d(0)];
      Eigen::Vector3d v1 = Vertices_3d[f.Vertices_3d(1)];
      Eigen::Vector3d v2 = Vertices_3d[f.Vertices_3d(2)];

      Eigen::Vector3d v0_uv =
          Eigen::Vector3d(Vertices_uv[f.Vertices_uv(0)](0),
                          Vertices_uv[f.Vertices_uv(0)](1), 0);
      Eigen::Vector3d v1_uv =
          Eigen::Vector3d(Vertices_uv[f.Vertices_uv(1)](0),
                          Vertices_uv[f.Vertices_uv(1)](1), 0);
      Eigen::Vector3d v2_uv =
          Eigen::Vector3d(Vertices_uv[f.Vertices_uv(2)](0),
                          Vertices_uv[f.Vertices_uv(2)](1), 0);

      Eigen::Matrix3d Ds;
      Ds.col(0) = v1 - v0;
      Ds.col(1) = v2 - v0;
      Ds.col(2) = f.F_normal;

      Eigen::Matrix3d Dm = Eigen::Matrix3d::Identity();
      Dm.col(0) = v1_uv - v0_uv;
      Dm.col(1) = v2_uv - v0_uv;
      Dm.col(2) = Dm.col(0).cross(Dm.col(1)).normalized();
      Eigen::Matrix3d inv_Dm = Dm.inverse();

      f.Deformation_gradient = Ds * inv_Dm;

      Eigen::JacobiSVD<Eigen::Matrix3d> svd;
      svd.compute(f.Deformation_gradient,
                  Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d U = svd.matrixU();
      Eigen::Matrix3d V = svd.matrixV();
      Eigen::Matrix3d VT = V.transpose();
      Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
      S.diagonal() = svd.singularValues();
      Eigen::Matrix3d R = U * VT;

      Eigen::Matrix3d L = Eigen::Matrix3d::Identity();
      L(2, 2) = R.determinant();
      S = S * L;
      double detU, detV;
      detU = U.determinant();
      detV = V.determinant();
      if (detU < 0 && detV > 0)
        U = U * L;
      if (detU > 0 && detV < 0) {
        V = V * L;
        VT = V.transpose();
      }
      R = U * VT;
      f.Sim = V * S * VT;
      if (abs(f.Sim(0, 1) - f.Sim(1, 0)) > 0.000001)
        std::cout << "pb" << std::endl;
      // f.Sim = f.Deformation_gradient;
    }

    return 0;
  }

  EPIXT fetch(const Eigen::Vector2d &uv) {
    // take fract part of uv mult by image size
    Eigen::Vector2d uvd = Eigen::Vector2d{
        -0.5 + (uv[0] - std::floor(uv[0])) * img_input_.width(),
        -0.5 + (uv[1] - std::floor(uv[1])) * img_input_.height()};
    Eigen::Vector2d uvfl{std::floor(uvd[0]), std::floor(uvd[1])};
    //  a = coef for linear interpolation
    Eigen::Vector2d a = uvd - uvfl;
    // c = integer texel coord
    Eigen::Vector2i c = uvfl.cast<int>();

    auto acces_repeat = [&](int xp, int yp) {
      int xx =
          xp < 0 ? img_input_.width() - 1 : (xp >= img_input_.width() ? 0 : xp);
      int yy = yp < 0 ? img_input_.height() - 1
                      : (yp >= img_input_.height() ? 0 : yp);
      return eigenPixel<double>(img_input_.pixelAbsolute(xx, yy));
    };

    return acces_repeat(c[0], c[1]);

    EPIXT V1 = (1.0 - a[0]) * acces_repeat(c[0], c[1]) +
               a[0] * acces_repeat(c[0] + 1, c[1]);
    EPIXT V2 = (1.0 - a[0]) * acces_repeat(c[0], c[1] + 1) +
               a[0] * acces_repeat(c[0] + 1, c[1] + 1);
    EPIXT V = (1.0 - a[1]) * V1 + a[1] * V2;

    return V;
  }

  EPIXT fetch_map(const Eigen::Vector2d &uv, const IMG &map) {
    // uv mult by map size
    Eigen::Vector2d pix_uv = Eigen::Vector2d{
        uv[0] * (map.width() - 1),
        uv[1] * (map.height() - 1)}; // uv in [0, 1], pix_uv in [0, 255]

    // partie entière et cast en int
    Eigen::Vector2d pix_floor{std::floor(pix_uv[0]),
                              std::floor(pix_uv[1])}; // pix_floor in {0, 255}
    Eigen::Vector2i ipix_floor = pix_floor.cast<int>();

    // partie décimale
    Eigen::Vector2d pix_fract = pix_uv - pix_floor; // pix_fract in [0, 1[

    // accès
    auto map_acces = [&](int xp, int yp) {
      return eigenPixel<double>(map.pixelAbsolute(xp, yp));
    };

    // interpolation bi-linéaire
    EPIXT map_interp_1 =
        (1. - pix_fract[0]) * map_acces(ipix_floor[0], ipix_floor[1]) +
        pix_fract[0] * map_acces(ipix_floor[0] + 1, ipix_floor[1]);
    EPIXT map_interp_2 =
        (1. - pix_fract[0]) * map_acces(ipix_floor[0], ipix_floor[1] + 1) +
        pix_fract[0] * map_acces(ipix_floor[0] + 1, ipix_floor[1] + 1);

    EPIXT map_interp =
        (1. - pix_fract[1]) * map_interp_1 + pix_fract[1] * map_interp_2;

    return map_interp;
  }

  PIXT tile_pixel(Eigen::Vector2d &uv) // fait le tnb pour un pixel
  {
    uv[1] = abs(uv[1] - 1.);
    //    grille
    Eigen::Vector3d B;
    Eigen::Vector2i vertex1, vertex2, vertex3;
    Face f = findTriangleContainingPoint(tree_, Vertices_uv, uv, 0);
    // TriangleGrid(f.Sim.template block<2, 2>(0, 0) * uv, B, vertex1, vertex2,
    // vertex3);
    TriangleGrid(uv, B, vertex1, vertex2, vertex3);

    // centers of tiles
    Eigen::Vector2d cen1 = MakeCenterUV(vertex1);
    Eigen::Vector2d cen2 = MakeCenterUV(vertex2);
    Eigen::Vector2d cen3 = MakeCenterUV(vertex3);

    Face f_v1 = findTriangleContainingPoint(tree_, Vertices_uv, cen1, 0);
    Face f_v2 = findTriangleContainingPoint(tree_, Vertices_uv, cen2, 0);
    Face f_v3 = findTriangleContainingPoint(tree_, Vertices_uv, cen3, 0);

    Eigen::Matrix2d M_transfo_v1 = f_v1.Sim.template block<2, 2>(0, 0);
    Eigen::Matrix2d M_transfo_v2 = f_v2.Sim.template block<2, 2>(0, 0);
    Eigen::Matrix2d M_transfo_v3 = f_v3.Sim.template block<2, 2>(0, 0);

    /*if (f_v1.area > 0.00001) {
      Eigen::Vector3d U_v1;
      barycentric(f_v1, cen1, U_v1.x(), U_v1.y(), U_v1.z());
      float dist_v1 = 1. - 3. * U_v1.minCoeff();
      dist_v1 = kernel_Normalize(dist_v1, 4.);

      M_transfo_v1 = (F_average_ * (1 - dist_v1) + dist_v1 * f_v1.Sim)
                         .template block<2, 2>(0, 0);
    }
    if (f_v2.area > 0.00001) {
      Eigen::Vector3d U_v2;
      barycentric(f_v2, cen2, U_v2.x(), U_v2.y(), U_v2.z());
      float dist_v2 = 1. - 3. * U_v2.minCoeff();
      dist_v2 = kernel_Normalize(dist_v2, 4.);

      M_transfo_v2 = (F_average_ * (1 - dist_v2) + dist_v2 * f_v2.Sim)
                         .template block<2, 2>(0, 0);
    }
    if (f_v3.area > 0.00001) {

      Eigen::Vector3d U_v3;
      barycentric(f_v3, cen3, U_v3.x(), U_v3.y(), U_v3.z());
      float dist_v3 = 1. - 3. * U_v3.minCoeff();
      dist_v3 = kernel_Normalize(dist_v3, 4.);

      M_transfo_v3 = (F_average_ * (1 - dist_v3) + dist_v3 * f_v3.Sim)
                         .template block<2, 2>(0, 0);
    }*/

    Eigen::Vector2d uv1 = M_transfo_v1 * (uv - cen1) + cen1 + hash(vertex1);
    Eigen::Vector2d uv2 = M_transfo_v2 * (uv - cen2) + cen2 + hash(vertex2);
    Eigen::Vector2d uv3 = M_transfo_v3 * (uv - cen3) + cen3 + hash(vertex3);
    /*uv1 = (uv - cen1) + cen1 + hash(vertex1);
    uv2 = (uv - cen2) + cen2 + hash(vertex2);
    uv3 = (uv - cen3) + cen3 + hash(vertex3);*/
    EPIXT t1 = fetch(uv1) - img_average_;
    EPIXT t2 = fetch(uv2) - img_average_;
    EPIXT t3 = fetch(uv3) - img_average_;

    auto W = B.normalized();

    EPIXT P = W[0] * t1 + W[1] * t2 + W[2] * t3 + img_average_;
    clamp(P);

    if (f.area > 0.0001) {

      Eigen::Vector3d U;
      barycentric(f, uv, U.x(), U.y(), U.z());
      float dist = 1. - 3. * U.minCoeff();
      dist = kernel_Normalize(dist, 4.);

      Eigen::Matrix3d F_tmp = F_average_ * (1 - dist) + dist * f.Sim;
      F_tmp = f.Sim;

      float a = F_tmp(0, 0);
      float b = F_tmp(1, 1);
      float c = F_tmp(0, 1);

      a = 0.5 * (atan(a) / (M_PI / 2.) + 1);
      b = 0.5 * (atan(b) / (M_PI / 2.) + 1);
      c = 0.5 * (atan(c) / (M_PI / 2.) + 1);

      // P = 255. * Eigen::Vector3d(a, b, c);
      //    P = Eigen::Vector3d(255, 0, 0);
    } else {
      Eigen::Matrix3d F_tmp = F_average_;

      float a = F_tmp(0, 0);
      float b = F_tmp(1, 1);
      float c = F_tmp(0, 1);

      a = 0.5 * (atan(a) / (M_PI / 2.) + 1);
      b = 0.5 * (atan(b) / (M_PI / 2.) + 1);
      c = 0.5 * (atan(c) / (M_PI / 2.) + 1);

      // P = 255. * Eigen::Vector3d(a, b, c);
      //    P = 255. * Eigen::Vector3d(0, 1, 0);
    }
    /*if (W[0] < 0.02)
      P = EPIXT(255., 255., 255.);
    if (W[1] < 0.02)
      P = EPIXT(0, 255., 255.);
    if (W[2] < 0.02)
      P = EPIXT(255., 255., 0.);*/

    return IMG::itkPixel(P);
  }

  void tile_img(
      IMG &img_out) // récupère le tnb pour chaque pixel de l'image de sortie
  {
    img_out.for_all_pixels([&](typename IMG::PixelType &P, int x, int y) {
      Eigen::Vector2d uv{double(x) / (img_out.width()),
                         double(y) / (img_out.height())};
      P = tile_pixel(uv);
    });
  }
};

template <typename IMG>
Tiling_n_Blending<IMG>
make_Tiling_n_Blending(const IMG &img,
                       const std::string &obj_file) // appel au constructeur
{
  return Tiling_n_Blending<IMG>(img, obj_file);
}

} // namespace ASTex
