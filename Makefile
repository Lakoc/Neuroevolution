PROJECT_ROOT:= /home/lakoc/FIT/1MIT/summer/EVO/Neuroevolution

.PHONY: run
run:
	python3 main.py

.PHONY: create_dris
create_dirs:
	cd $(PROJECT_ROOT)
	mkdir -p best fitness macs models params results

.PHONY: clean
clean:
	rm best/* fitness/* macs/* models/* params/* results/*