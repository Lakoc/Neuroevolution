PROJECT_ROOT:= /home/lakoc/FIT/1MIT/summer/EVO/Neuroevolution

.PHONY: run
run:
	python3 main.py

.PHONY: create_dirs
create_dirs:
	cd $(PROJECT_ROOT)
	mkdir -p best fitness macs models params results

.PHONY: arch
arch:
	zip -r xpolok03.zip src main.py Makefile xpolok03.pdf requirements.txt run_job.sh README.md

.PHONY: clean
clean:
	rm results/best/* results/fitness/* results/macs/* results/models/* results/params/* configs/* results/*