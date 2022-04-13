package services

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/WLM1ke/poptimizer/data/internal/bus"
	"github.com/WLM1ke/poptimizer/data/internal/domain"
	"github.com/WLM1ke/poptimizer/data/internal/repo"
	"github.com/WLM1ke/poptimizer/data/internal/rules/div/check"
	"github.com/WLM1ke/poptimizer/data/pkg/lgr"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

const _timeFormat = "2006-01-02"

// RawDivTableDTO представление информации о редактируемой таблице.
//
// Кроме данных таблицы хранит ID пользовательской сессии.
type RawDivTableDTO struct {
	SessionID string
	Ticker    string
	Rows      []domain.RawDiv
}

// NewRow шаблон для создания новой строки.
func (d RawDivTableDTO) NewRow() domain.RawDiv {
	if len(d.Rows) > 0 {
		return d.Rows[len(d.Rows)-1]
	}

	return domain.RawDiv{
		Date:     time.Now(),
		Value:    1,
		Currency: check.RUR,
	}
}

// RowDTO - представление добавляемой строки.
type RowDTO domain.RawDiv

// RawDivUpdate - сервис, обрабатывающая запросы по изменению таблицы с дивидендами.
//
// Позволяет загрузить существующую таблицу с данными и создать соответсвующую пользовательскую сессию по
// редактированию. Добавлять новые строки, сбрасывать и сохранять изменения в рамках пользовательской сессии.
type RawDivUpdate struct {
	logger *lgr.Logger
	repo   repo.ReadWrite[domain.RawDiv]

	lock     sync.Mutex
	tableDTO RawDivTableDTO

	bus *bus.EventBus
}

// NewRawDivEdit инициализирует сервис ручного ввода дивидендов.
func NewRawDivEdit(logger *lgr.Logger, db *mongo.Database, bus *bus.EventBus) *RawDivUpdate {
	return &RawDivUpdate{
		logger: logger,
		repo:   repo.NewMongo[domain.RawDiv](db),
		bus:    bus,
	}
}

// GetByTicker - возвращает сохраненные данные и создает пользовательскую сессию.
func (r *RawDivUpdate) GetByTicker(ctx context.Context, ticker string) (RawDivTableDTO, error) {
	table, err := r.repo.Get(ctx, domain.NewRawDivID(ticker))
	if err != nil {
		return RawDivTableDTO{}, fmt.Errorf(
			"can't load raw dividends from repo -> %w",
			err,
		)
	}

	r.lock.Lock()
	defer r.lock.Unlock()

	r.tableDTO = RawDivTableDTO{
		SessionID: primitive.NewObjectID().Hex(),
		Ticker:    ticker,
		Rows:      table.Rows(),
	}

	return r.tableDTO, nil
}

// AddRow добавляет новые строки в таблицу в рамках пользовательской сессии.
func (r *RawDivUpdate) AddRow(sessionID, date, value, currency string) (row RowDTO, err error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	if sessionID != r.tableDTO.SessionID {
		return row, fmt.Errorf(
			"wrong session id - %s",
			sessionID,
		)
	}

	row, err = parseRow(date, value, currency)
	if err != nil {
		return row, err
	}

	r.tableDTO.Rows = append(r.tableDTO.Rows, domain.RawDiv(row))

	return row, nil
}

func parseRow(date string, value string, currency string) (row RowDTO, err error) {
	row.Date, err = time.Parse(_timeFormat, date)
	if err != nil {
		return row, fmt.Errorf(
			"can't parse -> %w",
			err,
		)
	}

	row.Value, err = strconv.ParseFloat(value, 64)
	if err != nil {
		return row, fmt.Errorf(
			"can't parse -> %w",
			err,
		)
	}

	row.Currency = currency
	if currency != check.USD && currency != check.RUR {
		return row, fmt.Errorf(
			"incorrect currency - %s",
			currency,
		)
	}

	return row, nil
}

// Reload сбрасывает результаты редактирования в рамках пользовательской сессии и возвращает не измененную таблицу.
func (r *RawDivUpdate) Reload(ctx context.Context, sessionID string) (dto RawDivTableDTO, err error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	if sessionID != r.tableDTO.SessionID {
		return dto, fmt.Errorf(
			"wrong session id - %s",
			sessionID,
		)
	}

	table, err := r.repo.Get(ctx, domain.NewRawDivID(r.tableDTO.Ticker))
	if err != nil {
		return dto, fmt.Errorf(
			"can't load data from repo -> %w",
			err,
		)
	}

	r.tableDTO = RawDivTableDTO{
		SessionID: sessionID,
		Ticker:    string(table.Name()),
		Rows:      table.Rows(),
	}

	return r.tableDTO, nil
}

// Save сохраняет результаты редактирования.
func (r *RawDivUpdate) Save(ctx context.Context, sessionID string) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	defer func() { r.tableDTO.SessionID = "" }()

	if sessionID != r.tableDTO.SessionID {
		return fmt.Errorf(
			"wrong session id - %s",
			sessionID,
		)
	}

	tableID := domain.NewRawDivID(r.tableDTO.Ticker)

	rows := r.tableDTO.Rows
	sort.Slice(rows, func(i, j int) bool { return rows[i].Date.Before(rows[j].Date) })

	err := r.repo.Replace(ctx, domain.NewTable(tableID, time.Now(), rows))
	if err != nil {
		return fmt.Errorf("can't save to repo -> %w", err)
	}

	err = r.bus.Send(domain.NewUpdateCompleted(tableID))
	if err != nil {
		return fmt.Errorf("can't send update event -> %w", err)
	}

	return nil
}
