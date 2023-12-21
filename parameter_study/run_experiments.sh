touch parameter_study/data/log.txt

echo "$(date +%Y.%m.%d-%H:%M): Running random_init.py" >> parameter_study/data/log.txt
PYTHONPATH=. python3 parameter_study/experiments/random_init.py

echo "$(date +%Y.%m.%d-%H:%M): Running mask_function.py" >> parameter_study/data/log.txt
PYTHONPATH=. python3 parameter_study/experiments/mask_function.py

echo "$(date +%Y.%m.%d-%H:%M): Running depth.py" >> parameter_study/data/log.txt
PYTHONPATH=. python3 parameter_study/experiments/depth.py

echo "$(date +%Y.%m.%d-%H:%M): Finished" >> parameter_study/data/log.txt
echo "" >> parameter_study/log.txt
