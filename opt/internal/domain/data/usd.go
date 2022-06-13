package data

import (
	"context"
	"fmt"
	"time"

	"github.com/WLM1ke/gomoex"
	"github.com/WLM1ke/poptimizer/opt/internal/domain"
)

const (
	// _USDGroup группа и id данных курсе доллара.
	_USDGroup = "usd"

	_format = `2006-01-02`
	_ticker = `USD000UTSTOM`
)

// USDid - id котировок курса доллара.
func USDid() domain.QualifiedID {
	return domain.QualifiedID{
		Sub:   Subdomain,
		Group: _USDGroup,
		ID:    _USDGroup,
	}
}

// USD свечка с данными о курсе доллара.
type USD struct {
	Date     time.Time
	Open     float64
	Close    float64
	High     float64
	Low      float64
	Turnover float64
}

// TableUSD - таблица с котировками курса доллар.
type TableUSD = Table[USD]

// USDHandler обработчик событий, отвечающий за загрузку информации о курсе доллара.
type USDHandler struct {
	pub  domain.Publisher
	repo domain.ReadAppendRepo[TableUSD]
	iss  *gomoex.ISSClient
}

// NewUSDHandler создает обработчик событий, отвечающий за загрузку информации о курсе доллара.
func NewUSDHandler(
	pub domain.Publisher,
	repo domain.ReadAppendRepo[TableUSD],
	iss *gomoex.ISSClient,
) *USDHandler {
	return &USDHandler{
		iss:  iss,
		repo: repo,
		pub:  pub,
	}
}

// Match выбирает событие начала торгового дня.
func (h USDHandler) Match(event domain.Event) bool {
	return event.QualifiedID == TradingDateID() && event.Data == nil
}

func (h USDHandler) String() string {
	return "trading date -> usd"
}

// Handle реагирует на событие об обновлении торговых дат и обновляет курс.
func (h USDHandler) Handle(ctx context.Context, event domain.Event) {
	qid := USDid()

	event.QualifiedID = qid

	table, err := h.repo.Get(ctx, qid)
	if err != nil {
		event.Data = err
		h.pub.Publish(event)

		return
	}

	raw, err := h.download(ctx, event, table)
	if err != nil {
		event.Data = err
		h.pub.Publish(event)

		return
	}

	rows := h.convert(raw)

	if err := h.validate(table, rows); err != nil {
		event.Data = err
		h.pub.Publish(event)

		return
	}

	if !table.Entity.IsEmpty() {
		rows = rows[1:]
	}

	if rows.IsEmpty() {
		return
	}

	table.Timestamp = event.Timestamp
	table.Entity = rows

	if err := h.repo.Append(ctx, table); err != nil {
		event.Data = err
		h.pub.Publish(event)

		return
	}

	h.pub.Publish(event)
}

func (h USDHandler) download(
	ctx context.Context,
	event domain.Event,
	table domain.Aggregate[TableUSD],
) ([]gomoex.Candle, error) {
	start := ""
	if !table.Entity.IsEmpty() {
		start = table.Entity.LastRow().Date.Format(_format)
	}

	end := event.Timestamp.Format(_format)

	rowsRaw, err := h.iss.MarketCandles(
		ctx,
		gomoex.EngineCurrency,
		gomoex.MarketSelt,
		_ticker,
		start,
		end,
		gomoex.IntervalDay,
	)
	if err != nil {
		return nil, fmt.Errorf("can't download usd data -> %w", err)
	}

	return rowsRaw, nil
}

func (h USDHandler) convert(raw []gomoex.Candle) TableUSD {
	rows := make(TableUSD, 0, len(raw))

	for _, row := range raw {
		rows = append(rows, USD{
			Date:     row.Begin,
			Open:     row.Open,
			Close:    row.Close,
			High:     row.High,
			Low:      row.Low,
			Turnover: row.Value,
		})
	}

	return rows
}

func (h USDHandler) validate(table domain.Aggregate[TableUSD], rows TableUSD) error {
	prev := rows[0].Date
	for _, row := range rows[1:] {
		if prev.Before(row.Date) {
			prev = row.Date

			continue
		}

		return fmt.Errorf("not increasing dates %+v", prev)
	}

	if table.Entity.IsEmpty() {
		return nil
	}

	if table.Entity.LastRow() != rows[0] {
		return fmt.Errorf(
			"old rows %+v not match new %+v",
			table.Entity.LastRow(),
			rows[0])
	}

	return nil
}