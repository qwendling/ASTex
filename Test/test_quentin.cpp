#include <ASTex/image_gray.h>
#include <ASTex/image_rgb.h>

#include <ASTex/TilingBlending/tnb_from_obj.h>
#include <ASTex/easy_io.h>
#include <ASTex/special_io.h>

using namespace ASTex;

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cout << argv[0] << " img_example obj_file img_to_gen width height"
              << std::endl;
    return 1;
  }

  using IMGT = ImageRGBu8;
  IMGT img_ex;
  img_ex.load(std::string(argv[1]));

  int w = atoi(argv[4]);
  int h = atoi(argv[5]);
  IMGT img_out{w, h, false};

  auto tnb = make_Tiling_n_Blending(img_ex, std::string(argv[2]));

  tnb.tile_img(img_out);

  img_out.save(std::string(argv[3]));

  return EXIT_SUCCESS;
}
