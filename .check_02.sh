echo "---Проверка результатов модуля оценки функциональной значимости"

tree preprocessed_data/test

f="preprocessed_data/test/borzoi_with_cadd_gpn_msa.txt"; r=$(wc -l < "$f"); c=$(head -1 "$f" | awk -F'\t' '{print NF}'); printf "\n%d вариантам добавлено %d фичей\nв файл preprocessed_data/test/borzoi_with_cadd_gpn_msa.txt\n\n" "$((r-1))" "$((c-2))"
