// Command smoke runs the generated Go client against a live gateway.
// A client that only compiles proves nothing about the server.
//
//	go run ./cmd/smoke http://127.0.0.1:8000
package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	trigon "github.com/howlerops/trigon/sdk/go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: smoke <base url>")
		os.Exit(2)
	}
	ctx := context.Background()
	client := trigon.NewClient(os.Args[1])
	if key := os.Getenv("TRIGON_API_KEY"); key != "" {
		client.Headers["Authorization"] = "Bearer " + key
	}

	health, err := client.Healthz(ctx)
	must(err)
	fmt.Println("contract", trigon.ContractVersion, "| healthz", health["status"],
		"| calibrated", health["calibrated"])

	models, err := client.Models(ctx)
	must(err)
	fmt.Println("models  ", len(models), "tiers")

	resp, err := client.Decide(ctx,
		"my card was declined at the till and I still got charged",
		map[string]trigon.Question{
			"route": trigon.Choice("Which queue?",
				map[string]any{"name": "billing", "criteria": "a charge, refund or card payment"},
				map[string]any{"name": "shipping", "criteria": "a parcel or delivery"}),
			"severity": trigon.Score("How severe?",
				map[string]any{"name": "low", "value": 1.0},
				map[string]any{"name": "high", "value": 5.0}),
			"urgent": trigon.Noul("Needs a human within the hour?"),
		}, nil)
	must(err)
	route := resp.Answers["route"].Choice
	fmt.Println("model   ", resp.Model, "| tier", resp.Tier)
	fmt.Printf("route    %s conf %.3f\n", route.Selected, route.Confidence)
	fmt.Printf("severity %.3f\n", resp.Answers["severity"].Score.Score)
	fmt.Printf("urgent   p=%.3f\n", resp.Answers["urgent"].Noul.Probability)

	_, err = client.Decide(ctx, "x", map[string]trigon.Question{"bad": {"type": "nope"}}, nil)
	var apiErr *trigon.Error
	if !errors.As(err, &apiErr) {
		fmt.Fprintln(os.Stderr, "expected a typed API error, got", err)
		os.Exit(1)
	}
	fmt.Println("bad request ->", apiErr.Status)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
