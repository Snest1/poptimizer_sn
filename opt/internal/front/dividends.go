package front

import (
	"encoding/json"
	"net/http"

	"github.com/WLM1ke/poptimizer/opt/internal/domain/data/raw"
	"github.com/go-chi/chi"
)

func (f Frontend) registerDividendsHandlers() {
	f.mux.Get("/dividends", f.dividendsGetTickers)
	f.mux.Get("/dividends/{ticker}", f.dividendsGetDividends)
	f.mux.Put("/dividends/{ticker}", f.dividendsPutDividends)
}

func (f Frontend) dividendsGetTickers(writer http.ResponseWriter, request *http.Request) {
	dto, err := f.dividends.GetTickers(request.Context())
	if err != nil {
		err.Write(writer)

		return
	}

	writer.Header().Set("Content-Type", "application/json; charset=UTF-8")

	if err := json.NewEncoder(writer).Encode(dto); err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		f.logger.Warnf("can't encode tickers dto -> %s", err)

		return
	}
}

func (f Frontend) dividendsGetDividends(writer http.ResponseWriter, request *http.Request) {
	dto, err := f.dividends.GetDividends(request.Context(), chi.URLParam(request, "ticker"))
	if err != nil {
		err.Write(writer)

		return
	}

	writer.Header().Set("Content-Type", "application/json; charset=UTF-8")

	if err := json.NewEncoder(writer).Encode(dto); err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		f.logger.Warnf("can't encode dividends dto -> %s", err)

		return
	}
}

func (f Frontend) dividendsPutDividends(writer http.ResponseWriter, request *http.Request) {
	defer request.Body.Close()

	var dto raw.SaveDividendsDTO

	if err := json.NewDecoder(request.Body).Decode(&dto); err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		f.logger.Warnf("can't decode dividends dto -> %s", err)

		return
	}

	err := f.dividends.Save(request.Context(), dto)
	if err != nil {
		err.Write(writer)

		return
	}
}
