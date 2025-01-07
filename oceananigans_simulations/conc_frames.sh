# concatenates frames into one set of combined frames

frame_dir=("frames/frame_zeta" "frames/frame_b")

save_dir=conc
save_name=frame

export MAGICK_CONFIGURE_PATH=/local/mcrowe/imgk

for i in {0001..1201}
do
echo "combining frame $i ..."
convert ${frame_dir[0]}_$i.png ${frame_dir[1]}_$i.png +append ${save_dir}/${save_name}_$i.png
done