#!/bin/bash

TRAIN_DIR="data1/train"

echo "Починаємо перейменування файлів у $TRAIN_DIR"

for video_dir in "$TRAIN_DIR"/*/ ; do
    [ -d "$video_dir" ] || continue

    dir_name=$(basename "$video_dir")
    echo "Обробляємо: $dir_name"

    mapfile -t files < <(ls -v "$video_dir"/*.jpg 2>/dev/null)

    if [ ${#files[@]} -eq 0 ]; then
        echo "  Пропускаємо (немає jpg файлів)"
        continue
    fi

    echo "  Знайдено ${#files[@]} файлів — перейменовуємо..."

    counter=0
    for file in "${files[@]}"; do
        new_name=$(printf "%s/frame_%05d.jpg" "$video_dir" "$counter")

        if [ "$file" != "$new_name" ]; then
            mv "$file" "$new_name"
            echo "    $(basename "$file") → $(basename "$new_name")"
        fi

        ((counter++))
    done
done

echo "Готово! Усі папки оброблено."