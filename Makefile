# make              numbers and figures from the stored results, written to manuscript/
# make simulations  all experiments, which overwrite the stored results (about two hours)
# make clean        removes manuscript/

PYTHON ?= python3

.PHONY: all numbers figures simulations clean

all: numbers figures

numbers:
	cd code && $(PYTHON) make_numbers.py

figures:
	cd code && $(PYTHON) make_figures.py && $(PYTHON) make_fig4_cir_sweep.py && $(PYTHON) make_fig1_schematic.py

simulations:
	cd code && \
	$(PYTHON) run_design_points.py && \
	$(PYTHON) run_bep_hc.py && \
	$(PYTHON) run_bep_lc.py && \
	$(PYTHON) run_bep_lc_dense.py && \
	$(PYTHON) run_headline.py && \
	$(PYTHON) run_headline_dense.py && \
	$(PYTHON) run_tracking.py && \
	$(PYTHON) run_waveform.py && \
	$(PYTHON) run_waveform_isi.py && \
	$(PYTHON) check_isi_step.py && \
	$(PYTHON) run_design_rule.py && \
	$(PYTHON) run_zeta_crit.py && \
	$(PYTHON) run_bounded.py && \
	$(PYTHON) run_modification.py && \
	$(PYTHON) check_modification_bep.py && \
	$(PYTHON) run_leaky.py && \
	$(PYTHON) run_leaky_stochastic.py && \
	$(PYTHON) run_estimator_checks.py

clean:
	rm -rf manuscript code/__pycache__
