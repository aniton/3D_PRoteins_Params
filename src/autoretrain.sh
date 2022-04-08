PYTHONPATH=$(pwd)/src python3 -m solver \
--solver optimizer  \
--compare_method ssim \
--protein 1lmp \
--representation lines \
--style_image_path st.png

PYTHONPATH=$(pwd)/src python3 -m color_solver \
--protein 1lmp \
--representation lines \
--style_image_path st.png \
--params_txt texture_params.txt
