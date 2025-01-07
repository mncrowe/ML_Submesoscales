# creates a grid of frames from the four fields

frame_name=("frames/frame_zeta" "frames/frame_B" "frames/frame_w" "frames/frame_W")

save_name=frame
save_dir=conc_all

export MAGICK_CONFIGURE_PATH=/local/mcrowe/imgk

mkdir "$save_dir"

# loop through all frames
for n in {0001..3001}
do
  echo "combining frame $n ..."
  # create list of frames
  files_list=${frame_name[0]}"_"$n".png "${frame_name[1]}"_"$n".png "${frame_name[2]}"_"$n".png "${frame_name[3]}"_"$n".png"
  #echo "$files_list"
  montage -tile 2x2 -geometry +0+0 -border 0 $files_list ${save_dir}/${save_name}_$n.png   # -resize 1024

done