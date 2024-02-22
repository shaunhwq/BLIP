if __name__ == '__main__':
    import argparse
    import uvicorn
    from server import fast_api_app

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_address", type=str, help="IP address", default="0.0.0.0")
    parser.add_argument("--port_number", type=int, help="Flask app port number", default=4050)
    args = parser.parse_args()

    uvicorn.run(
        fast_api_app,
        host=args.ip_address,
        port=args.port_number,
    )
